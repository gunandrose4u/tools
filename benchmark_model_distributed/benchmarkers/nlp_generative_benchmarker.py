import datetime
import numpy as np
import threading

from abc import abstractmethod, ABC
from anubis_logger import logger
from time import perf_counter
from benchmarkers.benchmarker import DirectBenchmarker


class NlpGenerativeBenchmarker(DirectBenchmarker):
    def __init__(self, run_config, data_loader, backends, batch_size, metrics_analysis=None):
        super(NlpGenerativeBenchmarker, self).__init__(run_config, data_loader, backends, batch_size, metrics_analysis)

    def run(self):
        res_benchmark = super().run()

        token_predict_times = []
        for bk in self._backends:
            token_predict_times.extend(bk.token_predict_times)

        merged_token_predict_times = []
        for t_times in token_predict_times:
            if t_times:
                merged_token_predict_times.extend(t_times)


        if self._run_config.token_record:
            if self._run_config.verbose:
                logger.info(f"token_predict_times\n{merged_token_predict_times}")
                logger.info(np.mean(merged_token_predict_times))

            res_benchmark['token_predict_times'] = np.mean(merged_token_predict_times)
            res_benchmark['token_throughput'] = self._run_config.batch_size / np.mean(merged_token_predict_times)
        else:
            res_benchmark['predict_times'] = 'N/A'
            res_benchmark['throughput'] = 'N/A'

        return res_benchmark