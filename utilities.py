import argparse
import os

DEFAULT_METRICS_CSV_FILE = "metrics.csv"

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        choices=["bert-squad", "ssd-resnet34", "bart-base", "xlm-mlm-en-2048"],
        help="Model for benchmark",
    )

    parser.add_argument(
        "-mp",
        "--model_path",
        required=False,
        type=str,
        help="Model path for benchmark",
    )

    parser.add_argument(
        "-f",
        "--framework",
        required=True,
        type=str,
        default="onnxruntime",
        choices=["onnxruntime", "torch"],
        help="Framework to benchmark",
    )

    parser.add_argument(
        "-d",
        "--data",
        required=False,
        type=str,
        default="",
        help="Benchmark data folder, saved numpy array files",
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
        "-th",
        "--num_runner_threads",
        required=False,
        type=int,
        default=1,
        help="Threads to use for infernce",
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
        "--total_sample_count",
        required=False,
        default=-1,
        type=int,
        help="Batch size for inference",
    )

    parser.add_argument(
        "--warmup_times",
        required=False,
        default=20,
        type=int,
        help="Warmup times before calculate metrics of model",
    )

    parser.add_argument(
        "--same_batch",
        required=False,
        action="store_true",
        help="Run with same batch data")

    parser.add_argument(
        "--detect_batch_size",
        required=False,
        action="store_true",
        help="Detect batchsize to get max qps")

    parser.add_argument(
        "--detect_start_batch_size",
        required=False,
        default=1,
        type=int,
        help="The start batch size for detecting batchsize to get max qps",
    )

    parser.add_argument(
        "--detect_end_batch_size",
        required=False,
        default=6,
        type=int,
        help="The end batch size for detecting batchsize to get max qps",
    )

    parser.add_argument(
        "--device_ids",
        nargs="+",
        type=int,
        default=[0],
        help="Id of devices on which will be infernced, for each device id will a backend for it",
    )

    parser.add_argument(
        "--amd_gpu",
        required=False,
        action="store_true",
        help="Use AMD gpu")

    parser.add_argument(
        "--enable_queue",
        required=False,
        action="store_true",
        help="Threads pick up infernece query from queue")

    parser.add_argument(
        "-g",
        "--use_gpu",
        required=False,
        action="store_true",
        help="Run on gpu device")

    args = parser.parse_args()
    return args

def save_dict_to_csv(input_dict, csv_path=None):
    if not csv_path:
        csv_path = DEFAULT_METRICS_CSV_FILE

    dict = input_dict.copy()
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write(f"{','.join(dict.keys())}\n")

    for k in dict.keys():
        dict[k] = str(dict[k])

    with open(csv_path, 'a') as f:
        f.write(f"{','.join(dict.values())}\n")