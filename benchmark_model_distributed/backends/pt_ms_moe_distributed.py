import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import deepspeed
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.utils import is_offline_mode

from anubis_logger import logger
from pt_hf_nlp_distributed import HuggingFaceNlpGenerativeBackend

from megatron import get_args
from megatron import print_rank_0
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation_utils import get_token_stream


class MegatronHelper:
    """Helper class for Megatron-LM"""

    @staticmethod
    def add_text_generate_args(parser):
        """Text generation arguments."""
        group = parser.add_argument_group(title="text generation")

        group.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
        group.add_argument("--greedy", action="store_true", default=False, help="Use greedy sampling.")
        group.add_argument("--top_p", type=float, default=0.0, help="Top p sampling.")
        group.add_argument("--top_k", type=int, default=0, help="Top k sampling.")
        group.add_argument("--out-seq-length", type=int, default=None, help="Size of the output generated text.")
        group.add_argument(
            "--recompute",
            action="store_true",
            help="During generation recompute all attention " "instead of using previously computed keys/values.",
        )

        return parser

    @staticmethod
    def init_megatron(run_config):
        import sys

        if run_config.dtype == "float16":
            sys.argv += ("--fp16",)
        if run_config.greedy:
            sys.argv += ("--greedy",)
        sys.argv += ("--micro-batch-size", str(run_config.batch_size))
        sys.argv += ("--seq-length", str(run_config.seq_len))
        sys.argv += ("--out-seq-length", str(run_config.max_new_tokens))

        # Other arguments
        sys.argv += ("--tokenizer-type", "GPT2BPETokenizer", "--tensor-model-parallel-size", "1", "--num-layers", "24")
        sys.argv += ("--hidden-size", "1024", "--num-attention-heads", "16", "--max-position-embeddings", "1024")
        sys.argv += ("--num-experts", "1", "--mlp-type", "standard", "--temperature", "1.0")
        sys.argv += ("--top_p", "0.9", "--log-interval", "1", "--num-samples", "0", "--ds-inference")
        sys.argv += ("--vocab-file", "moe_data/gpt2-vocab.json")
        sys.argv += ("--merge-file", "moe_data/gpt2-merges.txt")
        sys.argv += ("--load", "moe_data/checkpoints/gpt2_345m", "--sample-input-file", "sample_input.txt")

        initialize_megatron(
            extra_args_provider=MegatronHelper.add_text_generate_args,
            args_defaults={"tokenizer_type": "GPT2BPETokenizer", "no_load_rng": True, "no_load_optim": True},
            ignore_unknown_args=True,
        )

    @staticmethod
    def model_provider(pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0("building GPT model ...")
        model = GPTModel(
            num_tokentypes=0,
            parallel_output=False,
            pre_process=pre_process,
            post_process=post_process,
            return_moe_loss=False,
        )  # we need to set "return_moe_loss" for the inference_mode
        return model

    @staticmethod
    def ds_inference(model, args):
        engine = deepspeed.init_inference(
            model=model,
            mp_size=args.tensor_model_parallel_size,
            tensor_parallel={"mpu": mpu},
            dtype=torch.half,
            replace_with_kernel_inject=True,
            moe_experts=args.num_experts,
            moe_type=args.mlp_type,
        )

        return engine.module


class MsMoeDeepSpeedBackend(HuggingFaceNlpGenerativeBackend):
    def __init__(self, run_config):
        super().__init__(run_config)

        deepspeed.init_distributed("nccl")
        self._rank = dist.get_rank()

        # since we're using deepspeed, set to False avoid get error as below:
        # AttributeError: 'DeepSpeedBloomInference' object has no attribute 'dtype'
        # It is introduced by enable torch.autocast
        self._amp_enabled = False

        if self._dtype != torch.float16:
            raise ValueError("Model microsoft/bloom-deepspeed-inference only supports fp16")

        MegatronHelper.init_megatron(run_config)
        self.megatron_args = get_args()
        if self.megatron_args.num_layers_per_virtual_pipeline_stage is not None:
            raise ValueError("Interleaved pipeline schedule is not yet supported for text generation.")

    def load_model(self):
        logger.info(f"Loading model {self._model_name}...")

        model = get_model(MegatronHelper.model_provider)
        if self.megatron_args.load is not None:
            load_checkpoint(model, None, None)
        self._model = model[0]

        if self.megatron_args.ds_inference:
            self._model = MegatronHelper.ds_inference(self._model, self.megatron_args)
            print("> DeepSpeed Inference engine initialized")

    def predict(self, input_tokens):
        if isinstance(input_tokens[0], int):  # batch_size == 1
            input_tokens = [input_tokens]
        for token_stream in get_token_stream(self._model, input_tokens):
            pass


class BenchmarkBackend(MsMoeDeepSpeedBackend):
    def __init__(self, run_config):
        super(BenchmarkBackend, self).__init__(run_config)
