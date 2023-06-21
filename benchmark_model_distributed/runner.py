import os

from utilities import parse_arguments, save_dict_to_csv, print_dict
from backends.backend import BackendFactory
from data_loaders.data_loader import DataLoaderFactory
from anubis_logger import logger
from supported_models import BENCHMARKER_MAPPING

class MultiBackendsRunner(object):
    def __init__(self, run_config, backend_nums=1):
        self._run_config = run_config
        self._backend_factory = BackendFactory()
        self._data_loader_factory = DataLoaderFactory()
        self._backends = []

        for _ in range(backend_nums):
            backend = self._backend_factory.get_backend(self._run_config)
            backend.load_model()
            self._backends.append(backend)

        self._data_loader = self._data_loader_factory.get_data_loader(self._run_config)

    def run(self):
        benchmarker = BENCHMARKER_MAPPING[self._run_config.benchmarker](self._run_config, self._data_loader, self._backends, self._run_config.batch_size)
        benchmarker.warmup()
        res_benchmark = benchmarker.run()
        if not self._run_config.distributed or self._run_config.local_rank == 0:
            res_benchmark.update(self._run_config.__dict__)
            print_dict("Benchmark result", res_benchmark)
            save_dict_to_csv(res_benchmark, self._run_config.result_csv)

class Runner(MultiBackendsRunner):
    def __init__(self, run_config):
        super().__init__(run_config, backend_nums=1)

def main():
    run_config = parse_arguments()

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    run_config.local_rank = local_rank
    run_config.world_size = world_size
    run_config.distributed = local_rank != -1 and world_size != -1

    print_dict("Run config", run_config.__dict__)

    runner = Runner(run_config)
    runner.run()

    logger.info("Benchmark done")

if __name__ == '__main__':
    main()