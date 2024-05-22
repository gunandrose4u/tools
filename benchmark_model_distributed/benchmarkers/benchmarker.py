import datetime
import numpy as np

from abc import abstractmethod, ABC
from anubis_logger import logger


class Benchmarker(ABC):
    def __init__(self, run_config, data_loader, backends, batch_size, metrics_analysis):
        self._data_loader = data_loader
        self._backends = backends
        self._metrics_analysis = metrics_analysis
        self._run_config = run_config
        self._batch_size = batch_size

        self._benchmark_start = None
        self._benchmark_end = None

        self._test_times = self._run_config.test_times

    def warmup(self):
        logger.info(f"Warmup model for {self._run_config.warmup_times} times")
        for i in range(self._run_config.warmup_times):
            for bk in self._backends:
                input = self._data_loader.get_batch_items(self._batch_size)
                if i == 0:
                    self._print_benchmark_input(input)
                bk.predict(input)

        logger.info(f"Warmup finished")

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def _print_benchmark_input(self, input):
        if isinstance(input, dict):
            for k in input:
                logger.info(f"Benchmark input data shape: {k}={input[k].shape}")
        if isinstance(input, list):  # For MoE case
            if isinstance(input[0], int):  # batch_size == 1
                logger.info(f"Benchmark input data shape: [{len(input)}]")
            else:
                logger.info(f"Benchmark input data shape: [{len(input)}, {len(input[0])}]")

    def _collect_metrics(self):
        predict_times = []
        for bk in self._backends:
            predict_times.extend(bk.predict_times)

        predict_times_in_ms = [ i * 1000 for i in predict_times]

        res_benchmark = {}
        res_benchmark['time'] = datetime.datetime.now().strftime("%m/%d/%Y %H:%M")
        res_benchmark['model'] = self._run_config.model
        res_benchmark['framework'] = f"{self._backends[0].name()}+{self._backends[0].version()}"
        res_benchmark['duration_sec'] = self._benchmark_end - self._benchmark_start
        self._get_percentile_metrics(res_benchmark, predict_times_in_ms, "query_latency_")

        return res_benchmark

    def _get_percentile_metrics(self, res_benchmark, raw_metrics, prefix=""):
        if res_benchmark is None:
            raise ValueError("res_benchmark is None")

        if raw_metrics:
            res_benchmark[f'{prefix}min'] = float(f"{np.min(raw_metrics):.3f}")
            res_benchmark[f'{prefix}max'] = float(f"{np.max(raw_metrics):.3f}")
            res_benchmark[f'{prefix}mean'] = float(f"{np.mean(raw_metrics):.3f}")
            res_benchmark[f'{prefix}50pt'] = float(f"{np.percentile(raw_metrics, 50):.3f}")
            res_benchmark[f'{prefix}90pt'] = float(f"{np.percentile(raw_metrics, 90):.3f}")
            res_benchmark[f'{prefix}95pt'] = float(f"{np.percentile(raw_metrics, 95):.3f}")
            res_benchmark[f'{prefix}99pt'] = float(f"{np.percentile(raw_metrics, 99):.3f}")
            res_benchmark[f'{prefix}99.9pt'] = float(f"{np.percentile(raw_metrics, 99.9):.3f}")
            res_benchmark[f'{prefix}var'] = float(f"{np.std(raw_metrics) / np.mean(raw_metrics):.3f}")
            res_benchmark[f'{prefix}std'] = float(f"{np.std(raw_metrics):.3f}")
        else:
            res_benchmark[f'{prefix}min'] = 0.0
            res_benchmark[f'{prefix}max'] = 0.0
            res_benchmark[f'{prefix}mean'] = 0.0
            res_benchmark[f'{prefix}50pt'] = 0.0
            res_benchmark[f'{prefix}90pt'] = 0.0
            res_benchmark[f'{prefix}95pt'] = 0.0
            res_benchmark[f'{prefix}99pt'] = 0.0
            res_benchmark[f'{prefix}99.9pt'] = 0.0
            res_benchmark[f'{prefix}var'] = 0.0
            res_benchmark[f'{prefix}std'] = 0.0
