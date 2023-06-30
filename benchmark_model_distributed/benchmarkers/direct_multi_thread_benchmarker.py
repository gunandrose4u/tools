import threading

from time import perf_counter

from anubis_logger import logger
from benchmarkers.direct_benchmarker import DirectBenchmarker

class DirectMultithreadBenchmarker(DirectBenchmarker):
    def __init__(self, run_config, data_loader, backends, batch_size, metrics_analysis=None):
        super(DirectMultithreadBenchmarker, self).__init__(run_config, data_loader, backends, batch_size, metrics_analysis)

    def run(self):
        logger.info(f"Start benchmarking {self._run_config.model} with {len(self._backends)} backends")
        runner_threads = []
        for bk_id in range(len(self._backends)):
            for _ in range(self._run_config.num_threads):
                t = threading.Thread(target=self._benchmark_with_backend, args=(bk_id, ))
                t.daemon = True
                runner_threads.append(t)

        self._benchmark_start = perf_counter()
        for t in runner_threads:
            t.start()

        for t in runner_threads:
            t.join()
        self._benchmark_end = perf_counter()
        logger.info(f"Benchmark finished")

        return self._collect_metrics()
