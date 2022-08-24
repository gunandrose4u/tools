import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=False,
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
        default=None,
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
        help="Threads to use",
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

    args = parser.parse_args()
    return args