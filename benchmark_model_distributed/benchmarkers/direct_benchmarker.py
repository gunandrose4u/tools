from time import perf_counter

from anubis_logger import logger
from benchmarkers.benchmarker import Benchmarker


class DirectBenchmarker(Benchmarker):
    def __init__(self, run_config, data_loader, backends, batch_size, metrics_analysis=None):
        super(DirectBenchmarker, self).__init__(run_config, data_loader, backends, batch_size, metrics_analysis)

        self._get_batch_feed = lambda : self._data_loader.get_batch_items(self._batch_size)

    def run(self):
        logger.info(f"Start benchmarking {self._run_config.model} with {len(self._backends)} backends")
        self._benchmark_start = perf_counter()
        self._benchmark_with_backend(0)
        self._benchmark_end = perf_counter()
        logger.info(f"Benchmark finished")

        return self._collect_metrics()

    def _benchmark_with_backend(self, bk_id):
        logger.info(f"Start benchmarking {self._run_config.model} for {self._test_times} times with number {bk_id} backend")
        for i in range(self._test_times):
            outputs = self._backends[bk_id].predict_with_perf(self._get_batch_feed())
            self._data_loader.post_process(outputs)

        logger.info(f"Benchmark with number {bk_id} backend finished")
