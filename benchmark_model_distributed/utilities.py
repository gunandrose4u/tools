import argparse
import os
import json

from anubis_logger import logger

DEFAULT_METRICS_CSV_FILE = "metrics.csv"

def run_config_validation(run_config):
    if run_config.test_times < 1:
        raise ValueError("test_times must be greater than 0")

    if run_config.backend_nums < 1:
        raise ValueError("backend_nums must be greater than 0")

    if run_config.batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    if run_config.num_runner_threads < 1:
        raise ValueError("num_runner_threads must be greater than 0")

    if run_config.warmup_times < 0:
        raise ValueError("warmup_times must be greater than or equal to 0")

    if run_config.seq_len < 1:
        raise ValueError("seq_len must be greater than 0")

    if run_config.max_tokens < 1:
        raise ValueError("max_tokens must be greater than 0")

    if run_config.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be greater than 0")

    if run_config.max_new_tokens > run_config.max_tokens:
        raise ValueError("max_new_tokens must be less than or equal to max_tokens")

    if run_config.num_beams < 1:
        raise ValueError("num_beams must be greater than 0")

    if run_config.model == "t5-3b":
        run_config.padding_side = "left"

    if run_config.model == "EleutherAI/gpt-j-6B":
        run_config.pad_token_id = 50256

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=False,
        type=str,
        default="facebook/opt-1.3b",
        choices=["facebook/opt-1.3b", "t5-3b", "EleutherAI/gpt-j-6B", "decapoda-research/llama-7b-hf", "decapoda-research/llama-13b-hf", "bigscience/bloom-7b1"],
        help="Model for benchmark, currently only support HuggingFace models",
    )

    parser.add_argument(
        "-bk",
        "--backend_name",
        required=False,
        default="pt_hf_nlp_distributed",
        type=str,
        help="Backend to benchmark, builtin backend is pt_hf_nlp_distributed. If you want benchmark model not supported yet, you just need implement your backend, put it under backends folder and pass the backend name here.",
    )

    parser.add_argument(
        "-d",
        "--dataloader",
        required=False,
        type=str,
        default="pt_hf_nlp",
        help="Dataloader to generate for benchmark, builtin dataloader is pt_hf_nlp. If you want benchmark model not supported yet, you just need implement your dataloader, put it under dataloaders folder and pass the dataloader name here.",
    )

    parser.add_argument(
        "-r",
        "--result_csv",
        required=False,
        default=DEFAULT_METRICS_CSV_FILE,
        help="CSV file for saving summary results.",
    )

    parser.add_argument(
        "-t",
        "--test_times",
        required=False,
        default=10,
        type=int,
        help="Number of repeat times to get average inference latency.",
    )

    parser.add_argument(
        "-n",
        "--num_threads",
        required=False,
        type=int,
        default=-1,
        help="Threads to use for framework",
    )

    parser.add_argument(
        "--num_runner_threads",
        required=False,
        type=int,
        default=1,
        help="Threads to use for inference on each backend",
    )

    parser.add_argument(
        "--backend_nums",
        required=False,
        type=int,
        default=1,
        help="Number of backends for inference",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        default=1,
        type=int,
        help="Batch size for inference",
    )

    parser.add_argument(
        "-s",
        "--seq_len",
        required=False,
        default=-1,
        type=int,
        help="Token length for inference",
    )

    parser.add_argument(
        "--dtype",
        required=False,
        default="float16",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="data-type for model to run",
    )

    parser.add_argument(
        "--padding_side",
        required=False,
        default="right",
        type=str,
        choices=["right", "left"],
        help="tokenizer padding side"
    )

    parser.add_argument(
        "--pad_token_id",
        required=False,
        default=-1,
        type=int,
        help="set pad token id for generation"
    )

    parser.add_argument(
        "--max_tokens",
        required=False,
        default=8200,
        type=int,
        help="maximum tokens used for the text-generation KV-cache"
    )

    parser.add_argument(
        "--max_new_tokens",
        required=False,
        default=32,
        type=int,
        help="maximum new tokens to generate"
    )

    parser.add_argument(
        "--use_kernel",
        required=False,
        action='store_true',
        help="enable kernel-injection for deepspeed"
    )

    parser.add_argument(
        "--greedy",
        required=False,
        action='store_true',
        help="greedy generation mode, it will override do_sample=False and num_beams=1"
    )

    parser.add_argument(
        "--do_sample",
        required=False,
        action='store_true',
        help="greedy generation mode"
    )

    parser.add_argument(
        "--num_beams",
        required=False,
        default=1,
        choices=[1, 4],
        type=int,
        help="beam search generation mode, set number of beams to search"
    )

    parser.add_argument(
        "--use_cache",
        required=False,
        action='store_true',
        help="use cache for generation"
    )

    parser.add_argument(
        "--benchmarker",
        required=False,
        type=str,
        default="nlp_generative",
        choices=["direct", "nlp_generative"],
        help="Benchmarker to benchmark",
    )

    parser.add_argument(
        "--warmup_times",
        required=False,
        default=3,
        type=int,
        help="Warmup times before calculate metrics of model",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        action='store_true'
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="""Please do not set this param, it is a dummy param.
        It is auto set when lunching runner.py by distributed launcher
        like torchrun, deepspeed, mpi, etc.
        When running in distributed mode, get local_rank from env vars."""
    )

    parser.add_argument(
        "--token_record",
        required=False,
        action='store_true',
        help="set True will record token-level latency for each inference in stopping criteria of HF generation method"
    )

    args = parser.parse_args()
    run_config_validation(args)
    return args

def save_dict_to_csv(input_dict, csv_path=DEFAULT_METRICS_CSV_FILE):
    dict = input_dict.copy()
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write(f"{','.join(dict.keys())}\n")

    for k in dict.keys():
        dict[k] = str(dict[k])

    with open(csv_path, 'a') as f:
        f.write(f"{','.join(dict.values())}\n")

def load_json_from_file(path):
    with open(path, 'r') as f:
        return json.load(f)

def print_dict(title, input_dict):
    logger.info(f"{title}:")
    for k, v in input_dict.items():
        logger.info(f"\t{k}={v}")
