import os
import sys
import traceback
import pathlib
import numpy as np
import threading

from utilities import parse_arguments, save_dict_to_csv
from backend import get_backend
from data_loader import FileDataLoader
from benchmarker import Benchmarker
from anubis_logger import logger

#BUILTIN_MODELS = ['bert-squad', 'bart-base', 'xlm-mlm-en-2048', 'ssd-resnet34']
FRAMEWORK_MAAPING = {"onnxruntime": "onnx", "torch": "torch"}
MODEL_FILETPYE_MAPPING = {"onnxruntime": [".onnx", ".ort"], "torch": [".bin", ".pytorch"]}


class Runner(object):
    def __init__(self, config, data_path, model_path, device_id):
        self._config = config
        self._data_path = data_path
        self._model_path = model_path
        self._device_id = device_id

        backend_opts = {}
        if self._config.num_threads > 0:
            backend_opts['num_threads'] = self._config.num_threads

        if self._config.use_gpu:
            backend_opts['use_gpu'] = self._config.use_gpu

            if self._device_id >= 0:
                backend_opts['device_id'] = self._device_id

        self._backend = get_backend(self._config.framework, backend_opts)
        self._backend.load(self._model_path)

        self._feeds = FileDataLoader(self._data_path)

    def run(self):
        if not self._config.detect_batch_size:
            self._run_with_batch(self._config.batch_size)
        else:
            detected_batch_sizes = []
            search_step = 8
            batch_size = self._config.detect_start_batch_size
            while(True):
                res_benchmark = self._run_with_batch(batch_size)
                detected_batch_sizes.append(res_benchmark['qps'])

                if batch_size >= self._config.detect_end_batch_size:
                    max_idx = np.argmax(detected_batch_sizes)
                    logger.info(f"Best batch size is {self._config.detect_start_batch_size + max_idx * search_step}, qps={detected_batch_sizes[max_idx]}")
                    logger.info(detected_batch_sizes)
                    break
                else:
                    batch_size += search_step
                    self._backend.clear_perf_details()

    def _run_with_batch(self, batch_size):
        benchmarker = Benchmarker(self._config, self._feeds, self._backend, batch_size)
        benchmarker.warmup()
        res_benchmark = benchmarker.run()
        logger.info(res_benchmark)
        save_dict_to_csv(res_benchmark, self._config.result_csv)
        return res_benchmark

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

    runner_threads = []
    for dv_id in args.device_ids:
        runner = Runner(args, data_path, model_path, dv_id)
        for i in range(args.num_runner_threads):
            t = threading.Thread(target=runner.run)
            t.daemon = True
            runner_threads.append(t)

    for t in runner_threads:
        t.start()

    for t in runner_threads:
        t.join()

    logger.info("Benchmark done")

if __name__ == '__main__':
    try:
        main()
    except:
        logger.error(traceback.format_exc())
        sys.exit(-1)