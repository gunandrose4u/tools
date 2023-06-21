import datetime
import numpy as np
import threading

from time import perf_counter

from anubis_logger import logger
from benchmarkers.benchmarker import Benchmarker


class DirectBenchmarker(Benchmarker):
    def __init__(self, run_config, data_loader, backends, batch_size, metrics_analysis=None):
        super(DirectBenchmarker, self).__init__(run_config, data_loader, backends, batch_size, metrics_analysis)

        self._get_batch_feed = lambda : self._data_loader.get_batch_items(self._batch_size)
        self._benchmark_start = None
        self._benchmark_end = None

    def run(self):
        logger.info(f"Start benchmarking {self._run_config.model} with {len(self._backends)} backends")
        self._benchmark_start = perf_counter()
        self._benchmark_with_backend(0)
        self._benchmark_end = perf_counter()
        logger.info(f"Benchmark finished")

        return self._collect_metrics()
    
    def _collect_metrics(self):
        predict_times = []
        for bk in self._backends:
            predict_times.extend(bk.predict_times)

        res_benchmark = {}
        res_benchmark['time'] = datetime.datetime.now().strftime("%m/%d/%Y %H:%M")
        res_benchmark['model'] = self._run_config.model
        res_benchmark['framework'] = f"{self._backends[0].name()}+{self._backends[0].version()}"
        res_benchmark['duration'] = self._benchmark_end - self._benchmark_start
        self._get_percentile_metrics(res_benchmark, predict_times, "query_latency_")

        return res_benchmark


    def _benchmark_with_backend(self, bk_id):
        logger.info(f"Start benchmarking {self._run_config.model} for {self._test_times} times with number {bk_id} backend")
        for i in range(self._test_times):
            outputs = self._backends[bk_id].predict_with_perf(self._get_batch_feed())
            self._data_loader.post_process(outputs)

        logger.info(f"Benchmark with number {bk_id} backend finished")

    def _get_percentile_metrics(self, res_benchmark, raw_metrics, prefix=""):
        res_benchmark[f'{prefix}min'] = np.min(raw_metrics)
        res_benchmark[f'{prefix}max'] = np.max(raw_metrics)
        res_benchmark[f'{prefix}mean'] = np.mean(raw_metrics)
        res_benchmark[f'{prefix}50pt'] = np.percentile(raw_metrics, 50)
        res_benchmark[f'{prefix}90pt'] = np.percentile(raw_metrics, 90)
        res_benchmark[f'{prefix}95pt'] = np.percentile(raw_metrics, 95)
        res_benchmark[f'{prefix}99pt'] = np.percentile(raw_metrics, 99)
        res_benchmark[f'{prefix}99.9pt'] = np.percentile(raw_metrics, 99.9)
        res_benchmark[f'{prefix}var'] = np.std(raw_metrics) / np.mean(raw_metrics)
        res_benchmark[f'{prefix}std'] = np.std(raw_metrics)
