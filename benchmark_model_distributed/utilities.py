import argparse
import os
import json

from anubis_logger import logger

DEFAULT_METRICS_CSV_FILE = "metrics.csv"

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=False,
        type=str,
        default="decapoda-research/llama-7b-hf",
        help="Model for benchmark",
    )

    parser.add_argument(
        "-bk",
        "--backend_name",
        required=False,
        default="pt_hf_nlp_distributed",
        type=str,
        help="Backend to benchmark",
    )

    parser.add_argument(
        "-d",
        "--dataloader",
        required=False,
        type=str,
        default="pt_hf_nlp",
        help="Dataloader to generate for benchmark",
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
        help="Seq length of text for inference",
    )

    parser.add_argument(
        "--dtype",
        required=False,
        default="float16",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="data-type"
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
        default=1024,
        type=int,
        help="maximum tokens used for the text-generation KV-cache"
    )

    parser.add_argument(
        "--max_new_tokens",
        required=False,
        default=50,
        type=int,
        help="maximum new tokens to generate"
    )

    parser.add_argument(
        "--use_kernel",
        required=False,
        action='store_true',
        help="enable kernel-injection"
    )

    parser.add_argument(
        "--greedy",
        required=False,
        action='store_true',
        help="greedy generation mode"
    )

    parser.add_argument(
        "--num_beams",
        required=False,
        default=1,
        type=int,
        help="beam search generation mode"
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
        default="direct",
        choices=["direct", "nlp_generative"],
        help="Benchmarker to benchmark",
    )

    parser.add_argument(
        "--warmup_times",
        required=False,
        default=5,
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
        help="local rank"
    )

    args = parser.parse_args()
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
