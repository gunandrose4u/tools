import datetime
import numpy as np

from anubis_logger import logger
from time import perf_counter


class Benchmarker(object):
    def __init__(self, config, data_loader, backend, batch_size, metrics_analysis=None):
        self._data_loader = data_loader
        self._backend = backend
        self._metrics_analysis = metrics_analysis
        self._config = config
        self._batch_size = batch_size if batch_size and batch_size > 1 else 1

        self._test_times = self._config.test_times
        if self._config.total_sample_count and self._config.total_sample_count >= self._batch_size:
            self._i_total_samples = self._config.total_sample_count
            self._test_times = int(self._i_total_samples / self._batch_size)
        else:
            self._i_total_samples = self._batch_size * self._test_times

        self._get_batch_feed = lambda : self._data_loader.get_item(self._batch_size)
        if self._config.same_batch:
            self._same_batch_feed = self._data_loader.get_item(self._batch_size)
            self._get_batch_feed = lambda : self._same_batch_feed

    def warmup(self):
        logger.info(f"Warmup model for {self._config.warmup_times} times")
        for i in range(self._config.warmup_times):
            self._backend.predict(self._get_batch_feed())
        logger.info(f"Warmup finished")

    def run(self):
        logger.info(f"Start benchmarking {self._config.model} for {self._test_times} times")
        t2_start = perf_counter()
        for i in range(self._test_times):
            self._backend.predict_with_perf(self._get_batch_feed())

        left_sample_cnt = self._i_total_samples % self._batch_size
        if left_sample_cnt > 0:
            self._test_times += 1
            self._backend.predict_with_perf(self._data_loader.get_item(left_sample_cnt))
        t2_end = perf_counter()
        logger.info(f"Benchmark finished")

        duration = t2_end - t2_start
        qps = self._i_total_samples / duration

        res_benchmark = {}
        res_benchmark['time'] = datetime.datetime.now().strftime("%m/%d/%Y %H:%M")
        res_benchmark['model'] = self._config.model
        res_benchmark['min'] = np.min(self._backend.predict_times)
        res_benchmark['max'] = np.max(self._backend.predict_times)
        res_benchmark['mean'] = np.mean(self._backend.predict_times)
        res_benchmark['50pt'] = np.percentile(self._backend.predict_times, 50)
        res_benchmark['90pt'] = np.percentile(self._backend.predict_times, 90)
        res_benchmark['95pt'] = np.percentile(self._backend.predict_times, 95)
        res_benchmark['99pt'] = np.percentile(self._backend.predict_times, 99)
        res_benchmark['99.9pt'] = np.percentile(self._backend.predict_times, 99.9)
        res_benchmark['var'] = np.std(self._backend.predict_times) / np.mean(self._backend.predict_times)
        res_benchmark['std'] = np.std(self._backend.predict_times)
        res_benchmark['qps'] = qps
        res_benchmark['batch_size'] = self._batch_size
        res_benchmark['total_samples'] = self._i_total_samples
        res_benchmark['framework'] = f"{self._config.framework}+{self._backend.version()}"
        res_benchmark['backend'] = self._backend.name()
        res_benchmark['test_times'] = self._test_times
        res_benchmark['warmup_times'] = self._config.warmup_times
        res_benchmark['num_threads'] = self._config.num_threads
        res_benchmark['use_gpu'] = self._config.use_gpu
        res_benchmark['duration'] = duration

        return res_benchmark

    def save_to_csv(self, dict, csv_path):
        logger.info(f"Save benchmark result for {self._config.model} to {csv_path}")
        if not csv_path:
            raise Exception("csv_path is not set")

        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write(f"{','.join(dict.keys())}\n")

        for k in dict.keys():
            dict[k] = str(dict[k])

        with open(csv_path, 'a') as f:
            f.write(f"{','.join(dict.values())}\n")