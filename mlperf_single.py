import array
import threading
import time

import os
import sys
import csv
import traceback
import pathlib
from turtle import back
import numpy as np
import datetime

from utilities import parse_arguments, save_dict_to_csv
from backend import get_backend
from data_loader import FileDataLoader
from anubis_logger import logger

import mlperf_loadgen as lg


backends = []
data_x = []

FRAMEWORK_MAAPING = {"onnxruntime": "onnx", "torch": "torch"}
MODEL_FILETPYE_MAPPING = {"onnxruntime": [".onnx"], "torch": [".bin", ".pytorch"]}


def load_samples_to_ram(query_samples):
    del query_samples
    return


def unload_samples_from_ram(query_samples):
    del query_samples
    return


def process_query_async(query_samples):
    """Processes the list of queries."""
    responses = []
    for s in query_samples:
        backends[0].predict_with_perf(data_x[s.index])
        responses.append(
            lg.QuerySampleResponse(s.id, 0, 0))
    lg.QuerySamplesComplete(responses)


def issue_query(query_samples):
    #threading.Thread(target=process_query_async,
    #    args=[query_samples]).start()
    process_query_async(query_samples)


def flush_queries():
    pass

def process_latencies(latencies_ns):
    pass


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

    backend_opts = {}
    if args.num_threads > 0:
        backend_opts['num_threads'] = args.num_threads
    backend = get_backend(args.framework, backend_opts)
    backend.load(model_path)

    backends.append(backend)

    feeds = FileDataLoader(data_path)
    for i in range(1024):
        data_x.append(feeds.get_item())

    for i in range(20):
        backend.predict(data_x[0])


    settings = lg.TestSettings()
    settings.scenario = lg.TestScenario.Offline
    settings.offline_expected_qps = 12
    #settings.scenario = lg.TestScenario.SingleStream
    settings.mode = lg.TestMode.PerformanceOnly
    #settings.single_stream_expected_latency_ns = 1000000
    #settings.min_query_count = args.test_times#100
    settings.min_duration_ms = 60000

    sut = lg.ConstructSUT(issue_query, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(
        1024, 512, load_samples_to_ram, unload_samples_from_ram)
    lg.StartTest(sut, qsl, settings)
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

    res_benchmark = {}
    res_benchmark['time'] = str(datetime.datetime.now())
    res_benchmark['model'] = args.model
    res_benchmark['min'] = str(np.min(backend.predict_times))
    res_benchmark['max'] = str(np.max(backend.predict_times))
    res_benchmark['mean'] = str(np.mean(backend.predict_times))
    res_benchmark['50pt'] = str(np.percentile(backend.predict_times, 50))
    res_benchmark['90pt'] = str(np.percentile(backend.predict_times, 90))
    res_benchmark['95pt'] = str(np.percentile(backend.predict_times, 95))
    res_benchmark['99pt'] = str(np.percentile(backend.predict_times, 99))
    res_benchmark['99.9pt'] = str(np.percentile(backend.predict_times, 99.9))
    res_benchmark['var'] = str(np.std(backend.predict_times) / np.mean(backend.predict_times))
    res_benchmark['std'] = str(np.std(backend.predict_times))
    res_benchmark['framework'] = f"{args.framework}+{backend.version()}"
    res_benchmark['backend'] = backend.name()
    res_benchmark['test_times'] = str(args.test_times)
    res_benchmark['num_threads'] = str(args.num_threads)

    logger.info(res_benchmark)
    logger.info("Benchmark done")

    save_dict_to_csv(res_benchmark, args.result_csv)


if __name__ == "__main__":
    main()
