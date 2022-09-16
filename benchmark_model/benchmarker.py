import datetime
import numpy as np
import threading

from abc import abstractmethod, ABC
from anubis_logger import logger
from time import perf_counter

class Benchmarker(ABC):
    def __init__(self, config, data_loader, backends, batch_size, metrics_analysis):
        self._data_loader = data_loader
        self._backends = backends
        self._metrics_analysis = metrics_analysis
        self._config = config
        self._batch_size = batch_size if batch_size and batch_size > 1 else 1

        self._test_times = self._config.test_times
        if self._config.total_sample_count and self._config.total_sample_count >= self._batch_size:
            self._i_total_samples = self._config.total_sample_count
            self._test_times = int(self._i_total_samples / self._batch_size)
        else:
            self._i_total_samples = self._batch_size * self._test_times

    def warmup(self):
        logger.info(f"Warmup model for {self._config.warmup_times} times")
        for i in range(self._config.warmup_times):
            for bk in self._backends:
                bk.predict(self._data_loader.get_batch_items(self._batch_size))
        logger.info(f"Warmup finished")

    @abstractmethod
    def run(self):
        raise NotImplementedError()

class DirectBenchmarker(Benchmarker):
    def __init__(self, config, data_loader, backends, batch_size, metrics_analysis=None):
        super(DirectBenchmarker, self).__init__(config, data_loader, backends, batch_size, metrics_analysis)

        self._get_batch_feed = lambda : self._data_loader.get_batch_items(self._batch_size)
        if self._config.same_batch:
            self._same_batch_feed = self._data_loader.get_batch_items(self._batch_size)
            self._get_batch_feed = lambda : self._same_batch_feed

    def run(self):
        logger.info(f"Start benchmarking {self._config.model} with {len(self._backends)} backends")
        runner_threads = []
        for bk_id in range(len(self._backends)):
            for i in range(self._config.num_runner_threads):
                t = threading.Thread(target=self._benchmark_with_backend, args=(bk_id, ))
                t.daemon = True
                runner_threads.append(t)

        t2_start = perf_counter()
        for t in runner_threads:
            t.start()

        for t in runner_threads:
            t.join()
        t2_end = perf_counter()
        logger.info(f"Benchmark finished")

        predict_times = []
        for bk in self._backends:
            predict_times.extend(bk.predict_times)

        res_benchmark = {}
        res_benchmark['time'] = datetime.datetime.now().strftime("%m/%d/%Y %H:%M")
        res_benchmark['model'] = self._config.model
        res_benchmark['min'] = np.min(predict_times)
        res_benchmark['max'] = np.max(predict_times)
        res_benchmark['mean'] = np.mean(predict_times)
        res_benchmark['50pt'] = np.percentile(predict_times, 50)
        res_benchmark['90pt'] = np.percentile(predict_times, 90)
        res_benchmark['95pt'] = np.percentile(predict_times, 95)
        res_benchmark['99pt'] = np.percentile(predict_times, 99)
        res_benchmark['99.9pt'] = np.percentile(predict_times, 99.9)
        res_benchmark['var'] = np.std(predict_times) / np.mean(predict_times)
        res_benchmark['std'] = np.std(predict_times)
        res_benchmark['qps'] = ((self._i_total_samples * len(self._backends)) / np.sum(predict_times)) * len(self._backends)
        res_benchmark['batch_size'] = self._batch_size
        res_benchmark['total_samples'] = self._i_total_samples * len(self._backends)
        res_benchmark['framework'] = f"{self._config.framework}+{self._backends[0].version()}"
        res_benchmark['backend'] = self._backends[0].name()
        res_benchmark['test_times'] = self._test_times
        res_benchmark['warmup_times'] = self._config.warmup_times
        res_benchmark['num_threads'] = self._config.num_threads
        res_benchmark['use_gpu'] = self._config.use_gpu
        res_benchmark['duration'] = t2_end - t2_start

        return res_benchmark

    def _benchmark_with_backend(self, bk_id):
        logger.info(f"Start benchmarking {self._config.model} for {self._test_times} times with number {bk_id} backend")
        for i in range(self._test_times):
            self._backends[bk_id].predict_with_perf(self._get_batch_feed())

        left_sample_cnt = self._i_total_samples % self._batch_size
        if left_sample_cnt > 0:
            self._test_times += 1
            self._backends[bk_id].predict_with_perf(self._data_loader.get_batch_items(left_sample_cnt))
        logger.info(f"Benchmark with number {bk_id} backend finished")
