import os
import sys
import traceback
import pathlib
import numpy as np
import threading

from utilities import parse_arguments, save_dict_to_csv
from backend import BackendFactory
from data_loader import FileDataLoader, DataLoaderFactory
from benchmarker import DirectBenchmarker
from mlperf_benchmarker import MlPerfBenchmarker
from anubis_logger import logger

#BUILTIN_MODELS = ['bert-squad', 'bart-base', 'xlm-mlm-en-2048', 'ssd-resnet34']
FRAMEWORK_MAAPING = {"onnxruntime": "onnx", "torch": "torch"}
MODEL_FILETPYE_MAPPING = {"onnxruntime": [".onnx", ".ort"], "torch": [".bin", ".pytorch"]}
BENCHMARKER_MAPPING = {"direct": DirectBenchmarker, "mlperf": MlPerfBenchmarker}


class Runner(object):
    def __init__(self, config, data_path, model_path, backend_factory=None, data_loader_factory=None):
        self._config = config
        self._data_path = data_path
        self._model_path = model_path
        self._backend_factory = backend_factory if backend_factory else BackendFactory()
        self._data_loader_factory = data_loader_factory if data_loader_factory else DataLoaderFactory()

        self._backends = []
        for dv_id in self._config.device_ids:
            backend = self._backend_factory.get_backend(self._config, dv_id)
            backend.load(self._model_path)
            self._backends.append(backend)

        self._feeds = self._data_loader_factory.get_data_loader(self._config, self._data_path)

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
        benchmarker = BENCHMARKER_MAPPING[self._config.benchmarker](self._config, self._feeds, self._backends, batch_size)
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

    runner = Runner(args, data_path, model_path)
    runner.run()

    logger.info("Benchmark done")

if __name__ == '__main__':
    try:
        main()
    except:
        logger.error(traceback.format_exc())
        sys.exit(-1)