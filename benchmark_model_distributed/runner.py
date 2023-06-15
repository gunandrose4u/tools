import os
import sys
import traceback
import pathlib
import numpy as np
import threading

from utilities import parse_arguments, save_dict_to_csv, print_dict
from backends.backend import BackendFactory
from data_loaders.data_loader import DataLoaderFactory
from benchmarkers.benchmarker import DirectBenchmarker
from benchmarkers.nlp_generative_benchmarker import NlpGenerativeBenchmarker
from anubis_logger import logger

BENCHMARKER_MAPPING = {
    "direct": DirectBenchmarker,
    "nlp_generative": NlpGenerativeBenchmarker,
}

class Runner(object):
    def __init__(self, run_config):
        self._run_config = run_config
        self._backend_factory = BackendFactory()
        self._data_loader_factory = DataLoaderFactory()

        self._backends = []
        for _ in range(self._run_config.backend_nums):
            backend = self._backend_factory.get_backend(self._run_config)
            backend.load_model()
            self._backends.append(backend)

        self._data_loader = self._data_loader_factory.get_data_loader(self._run_config)

    def run(self):
        benchmarker = BENCHMARKER_MAPPING[self._run_config.benchmarker](self._run_config, self._data_loader, self._backends, self._run_config.batch_size)
        benchmarker.warmup()
        res_benchmark = benchmarker.run()
        print_dict("Benchmark result", res_benchmark)
        res_benchmark.update(self._run_config.__dict__)
        save_dict_to_csv(res_benchmark, self._run_config.result_csv)

def main():
    run_config = parse_arguments()

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    if local_rank != -1 and world_size != -1:
        run_config.local_rank = local_rank
        run_config.world_size = world_size
        run_config.distributed = True
    else:
        run_config.distributed = False

    print_dict("Run config", run_config.__dict__)

    runner = Runner(run_config)
    runner.run()

    logger.info("Benchmark done")

if __name__ == '__main__':
    main()