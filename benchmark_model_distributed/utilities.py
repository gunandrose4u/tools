import argparse
import pathlib
import os
import json

from anubis_logger import logger
from supported_models import SUPPORTED_MODELS, BENCHMARKER_MAPPING
from benchmarkers.mlperf.const import SINGLESTREAM, OFFLINE


DEFAULT_METRICS_CSV_FILE = "metrics.csv"

def list_built_in_modules(path_to_modules, exclude_modules=[]):
    return [item[:-3] for item in os.listdir(path_to_modules)
            if not os.path.isdir(os.path.join(path_to_modules, item)) and item.endswith(".py") and item != "__init__.py" and item not in exclude_modules]

def run_config_validation(run_config):
    if run_config.test_times < 1:
        raise ValueError("test_times must be greater than 0")

    if run_config.batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    if run_config.warmup_times < 0:
        raise ValueError("warmup_times must be greater than or equal to 0")

    if run_config.seq_len < 1:
        raise ValueError("seq_len must be greater than 0")

    if run_config.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be greater than 0")

    if run_config.num_beams < 1:
        raise ValueError("num_beams must be greater than 0")

    if run_config.model == "t5-3b":
        run_config.padding_side = "left"

    if run_config.model == "EleutherAI/gpt-j-6B":
        run_config.pad_token_id = 50256

    if not run_config.backend_name:
        run_config.backend_name = SUPPORTED_MODELS[run_config.model][0]

    if not run_config.dataloader:
        run_config.dataloader = SUPPORTED_MODELS[run_config.model][1]

    if not run_config.benchmarker:
        run_config.benchmarker = SUPPORTED_MODELS[run_config.model][2]

def parse_arguments():
    parser = argparse.ArgumentParser()

    curdir = pathlib.Path(__file__).parent.resolve()
    BUILTIN_BACKENDS = list_built_in_modules(os.path.join(curdir, 'backends'), exclude_modules=['backend.py'])
    BUILTIN_DATALOADERS = list_built_in_modules(os.path.join(curdir, 'data_loaders'), exclude_modules=['data_loader.py'])

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        type=str,
        choices=SUPPORTED_MODELS.keys(),
        help="Model to benchmark, currently only support HuggingFace models",
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        required=False,
        default=1,
        type=int,
        help="Batch size for inference",
    )

    parser.add_argument(
        "--seq_len",
        "-s",
        required=False,
        default=16,
        type=int,
        help="Token length for inference",
    )

    parser.add_argument(
        "--max_new_tokens",
        required=False,
        default=16,
        type=int,
        help="""The maximum numbers of tokens to generate,
        ignoring the number of tokens in the prompt."""
    )

    parser.add_argument(
        "--do_sample",
        required=False,
        action='store_true',
        help="Whether or not to use sampling ; use greedy decoding otherwise"
    )

    parser.add_argument(
        "--num_beams",
        required=False,
        default=1,
        choices=[1, 4],
        type=int,
        help="Number of beams for beam search. 1 means no beam search."
    )

    parser.add_argument(
        "--greedy",
        required=False,
        action='store_true',
        help="Greedy generation mode, it will override do_sample=False and num_beams=1"
    )

    parser.add_argument(
        "--use_cache",
        required=False,
        action='store_true',
        help="""Whether or not the model should use the past last key/values
        attentions (if applicable to the model) to speed up decoding."""
    )

    parser.add_argument(
        "--use_kernel",
        required=False,
        action='store_true',
        help="Enable kernel-injection for deepspeed distributed inference"
    )

    parser.add_argument(
        "--dtype",
        required=False,
        default="float16",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model to run",
    )

    parser.add_argument(
        "--padding_side",
        required=False,
        default="right",
        type=str,
        choices=["right", "left"],
        help="Tokenizer padding side"
    )

    parser.add_argument(
        "--pad_token_id",
        required=False,
        default=-1,
        type=int,
        help="Set pad token id for generation"
    )

    parser.add_argument(
        "--num_threads",
        "-n",
        required=False,
        type=int,
        default=-1,
        help="Number of threads to use for framework",
    )

    parser.add_argument(
        "--warmup_times",
        required=False,
        default=3,
        type=int,
        help="Warmup times before calculate metrics of model",
    )

    parser.add_argument(
        "--test_times",
        "-t",
        required=False,
        default=10,
        type=int,
        help="Number of repeat times to predict with model to get benchmark metrics.",
    )

    parser.add_argument(
        "--total_sample_count",
        required=False,
        default=10,
        type=int,
        help="""Total count of input dataset for benchmarking, each time
        benchmark will randomly select one data from the whole dataset""",
    )

    parser.add_argument(
        "--benchmarker",
        required=False,
        type=str,
        choices=BENCHMARKER_MAPPING.keys(),
        help="Benchmarker to benchmark",
    )

    parser.add_argument(
        "--backend_name",
        "-bk",
        required=False,
        choices=BUILTIN_BACKENDS,
        type=str,
        help="""Backend to benchmark, builtin backend is pt_hf_nlp_distributed.
        If a model is not supported yet, you just need implement your own backend
        to support the model, and put it under backends folder, pass the backend
        name here.""",
    )

    parser.add_argument(
        "--dataloader",
        "-d",
        required=False,
        type=str,
        choices=BUILTIN_DATALOADERS,
        help="""Dataloader to generate for benchmark, builtin dataloader is pt_hf_nlp.
        If current dataloader is not fit for you, you just need implement your own
        dataloader, put it under dataloaders folder and pass the dataloader name here.""",
    )

    parser.add_argument(
        "--token_metrics",
        required=False,
        action='store_true',
        help="""Set True will record token-level latency for each inference
        in stopping criteria of HF generation method"""
    )

    parser.add_argument(
        "--verbose",
        "-v",
        required=False,
        action='store_true'
    )

    parser.add_argument(
        "--result_csv",
        "-r",
        required=False,
        default=DEFAULT_METRICS_CSV_FILE,
        help="""Csv file for saving summary results. If the csv file exists,
        the results will be appended to the file.""",
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

    # mlperf params
    if "mlperf" in BENCHMARKER_MAPPING.keys():
        parser.add_argument(
            "--mlperf_scenario",
            required=False,
            type=str,
            default=SINGLESTREAM,
            choices=[SINGLESTREAM, OFFLINE],
            help="Mlperf scenario",
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
