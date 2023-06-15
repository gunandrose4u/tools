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

        first_token_predict_times = []
        non_first_token_predict_times = []
        for t_times in token_predict_times:
            num_new_tokens = len(t_times)
            if num_new_tokens > 0:
                first_token_predict_times.append(t_times[0])
            
            if num_new_tokens > 1:
                non_first_token_predict_times.extend(t_times[1:])

        if self._run_config.verbose:
            logger.info(f"token_predict_times\n{token_predict_times}")
            logger.info(f"first_token_predict_times\n{first_token_predict_times}")
            logger.info(f"non_first_token_predict_times\n{non_first_token_predict_times}")
            logger.info(np.mean(first_token_predict_times))
            logger.info(np.mean(non_first_token_predict_times))
        
        res_benchmark['first_token_mean_latency'] = np.mean(first_token_predict_times)
        res_benchmark['token_mean_latency'] = np.mean(non_first_token_predict_times)
        res_benchmark['first_token_throughput'] = self._run_config.batch_size / np.mean(first_token_predict_times)
        res_benchmark['token_throughput'] = self._run_config.batch_size / np.mean(non_first_token_predict_times)
        

        return res_benchmark