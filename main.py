
from ast import arg
import os
from random import random
import sys
import csv
import traceback
import pathlib
import numpy as np
import datetime
import random
import time

from utilities import parse_arguments, save_dict_to_csv
from backend import get_backend
from data_loader import FileDataLoader
from anubis_logger import logger

#BUILTIN_MODELS = ['bert-squad', 'bart-base', 'xlm-mlm-en-2048', 'ssd-resnet34']
FRAMEWORK_MAAPING = {"onnxruntime": "onnx", "torch": "torch"}
MODEL_FILETPYE_MAPPING = {"onnxruntime": [".onnx"], "torch": [".bin", ".pytorch"]}

def main():
    try:
        args = parse_arguments()
        logger.info(args)
    except:
        return

    curdir = pathlib.Path(__file__).parent.resolve()
    models_dir = os.path.join(curdir, 'models')
    data_path = args.data
    model_path = args.model_path
    BUILTIN_MODELS = [item for item in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, item))]
    logger.info(f"BUILTIN_MODELS are {BUILTIN_MODELS}")


    if args.model:
        if args.model in BUILTIN_MODELS:
            model_folder = os.path.join(models_dir, args.model, FRAMEWORK_MAAPING[args.framework])
            data_path = model_folder
            model_path = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if os.path.splitext(f)[1] in MODEL_FILETPYE_MAPPING[args.framework]][0]
        else:
            logger.error(f"{args.model} is not in builtin models {BUILTIN_MODELS}")
            return
    else:
        if not args.model_path or not args.data_path:
            logger.error(f"Need set model path and data path when benchmark model is not a builtin model")
            return

    backend_opts = {}
    if args.num_threads > 0:
        backend_opts['num_threads'] = args.num_threads

    if args.use_gpu:
        backend_opts['use_gpu'] = args.use_gpu

    backend = get_backend(args.framework, backend_opts)
    backend.load(model_path)

    feeds = FileDataLoader(data_path)

    batch_size = 1
    if args.batch_size and args.batch_size > 1:
        batch_size = args.batch_size

    test_times = args.test_times
    if args.total_sample_count and args.total_sample_count >= batch_size:
        i_total_samples = args.total_sample_count
        test_times = int(i_total_samples / batch_size)
    else:
        i_total_samples = batch_size * test_times

    for i in range(args.warmup_times):
        backend.predict(feeds.get_item(batch_size))

    batch_feed = lambda : feeds.get_item(batch_size)
    if args.same_batch:
        tmp = feeds.get_item(batch_size)
        batch_feed = lambda : tmp

    t2_start = time.perf_counter()
    for i in range(test_times):
        backend.predict_with_perf(batch_feed())

    left_sample_cnt = i_total_samples % batch_size
    if left_sample_cnt > 0:
        test_times += 1
        backend.predict_with_perf(feeds.get_item(left_sample_cnt))
    t2_end = time.perf_counter()
    qps = i_total_samples / (t2_end - t2_start)

    res_benchmark = {}
    res_benchmark['time'] = datetime.datetime.now().strftime("%m/%d/%Y %H:%M")
    res_benchmark['model'] = args.model
    res_benchmark['min'] = np.min(backend.predict_times)
    res_benchmark['max'] = np.max(backend.predict_times)
    res_benchmark['mean'] = np.mean(backend.predict_times)
    res_benchmark['50pt'] = np.percentile(backend.predict_times, 50)
    res_benchmark['90pt'] = np.percentile(backend.predict_times, 90)
    res_benchmark['95pt'] = np.percentile(backend.predict_times, 95)
    res_benchmark['99pt'] = np.percentile(backend.predict_times, 99)
    res_benchmark['99.9pt'] = np.percentile(backend.predict_times, 99.9)
    res_benchmark['var'] = np.std(backend.predict_times) / np.mean(backend.predict_times)
    res_benchmark['std'] = np.std(backend.predict_times)
    res_benchmark['qps'] = qps
    res_benchmark['batch_size'] = batch_size
    res_benchmark['total_samples'] = i_total_samples
    res_benchmark['framework'] = f"{args.framework}+{backend.version()}"
    res_benchmark['backend'] = backend.name()
    res_benchmark['test_times'] = test_times
    res_benchmark['warmup_times'] = args.warmup_times
    res_benchmark['num_threads'] = args.num_threads
    res_benchmark['use_gpu'] = args.use_gpu
    res_benchmark['duration'] = t2_end - t2_start

    logger.info(res_benchmark)
    logger.info("Benchmark done")

    save_dict_to_csv(res_benchmark, args.result_csv)

if __name__ == '__main__':
    try:
        main()
    except:
        logger.error(traceback.format_exc())
        sys.exit(-1)