# usage:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom
#
# to run benchmarks:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom --benchmark
#


# This is going to improve, but at the moment, the process is a bit cumbersome - we first use
# 1. use Deepspeed-ZeRO to instantiate the model on GPUs, w/o loading the checkpoints,
# 2. free the allocated storage
# 3. start Deepspeed-Inference and only now load the checkpoint
# 4. run generate
# Done.
#


# This file is from https://github.com/huggingface/transformers-bloom-inference/blob/main/bloom-inference-scripts/bloom-ds-inference.py#L69


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


class MsBloomDeepSpeedBackend(HuggingFaceNlpGenerativeBackend):
    def __init__(self, run_config):
        super(MsBloomDeepSpeedBackend, self).__init__(run_config)
        self._tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8", "microsoft/bloom-deepspeed-inference-fp16"]
        self._tp_presharded_mode = True if self._model_name in self._tp_presharded_models else False

        deepspeed.init_distributed("nccl")
        self._rank = dist.get_rank()

        # since we're using deepspeed, set to False avoid get error as below:
        # AttributeError: 'DeepSpeedBloomInference' object has no attribute 'dtype'
        # It is introduced by enable torch.autocast
        self._amp_enabled = False

        if not run_config.distributed:
            raise ValueError("Model microsoft/bloom-deepspeed-inference inference only works with distributed mode")

    def load_model(self):
        logger.info(f"Loading model {self._model_name}...")

        config = AutoConfig.from_pretrained(self._model_name)
        # XXX: can't automatically derive dtype via config's `from_pretrained`
        # dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16


        # use one of these args to `init_inference`
        # 1. injection_policy is the slower version, but it's plain pytorch so it'll always work
        # 2. replace_with_self._run_config.use_kernel is the faster one (fast fused kernels)
        if self._run_config.use_kernel:
            # XXX: for now ds-inference only works with fp16
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        self._clear_memory("pre-from-pretrained")
        # Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
        with deepspeed.OnDevice(dtype=dtype, device="meta"):
            self._model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

        deepspeed.runtime.utils.see_memory_usage("post-from-pretrained", force=True)

        self._model = self._model.eval()

        self._clear_memory("post-init-ds-zero-init")

        checkpoints_json = "checkpoints.json"

        self._clear_memory("pre-ds-inference-init")

        if self._run_config.use_kernel:
            kwargs = dict(replace_with_kernel_inject=True)
        else:
            kwargs = dict(injection_policy={BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")})

        repo_root = self._get_repo_root(self._model_name)
        if self._tp_presharded_mode:
            # tp presharded repos come with their own checkpoints config file
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        else:
            # for normal bloom repo we need to write the checkpoints config file
            self._write_checkpoints_json(checkpoints_json)
            dist.barrier()

        # checkpoints_json=None
        self._model = deepspeed.init_inference(
            self._model,
            mp_size=self._run_config.world_size,
            base_dir=repo_root,
            dtype=self._dtype,
            checkpoint=checkpoints_json,
            **kwargs,
        )

        self._clear_memory("post-ds-inference-init")

        self._model = self._model.module


    def _clear_memory(self, title):
        torch.cuda.empty_cache()
        gc.collect()
        deepspeed.runtime.utils.see_memory_usage(title, force=True)

    def _get_repo_root(self, model_name_or_path):
        # checks if online or not
        if is_offline_mode():
            logger.info("Offline mode: forcing local_files_only=True")

        # download only on first process
        if self._rank == 0:
            snapshot_download(
                model_name_or_path,
                local_files_only=is_offline_mode(),
                cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                ignore_patterns=["*.safetensors"],
            )

        dist.barrier()

        return snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            ignore_patterns=["*.safetensors"],
        )


    def _get_checkpoint_files(self, model_name_or_path):
        cached_repo_dir = self._get_repo_root(model_name_or_path)

        # extensions: .bin | .pt
        # creates a list of paths from all downloaded files in cache dir
        file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
        return file_list

    def _write_checkpoints_json(self, checkpoints_json):
        checkpoint_files = self._get_checkpoint_files(self._model_name)
        if self._rank == 0:
            data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}
            json.dump(data, open(checkpoints_json, "w"))


class BenchmarkBackend(MsBloomDeepSpeedBackend):
    def __init__(self, run_config):
        super(BenchmarkBackend, self).__init__(run_config)