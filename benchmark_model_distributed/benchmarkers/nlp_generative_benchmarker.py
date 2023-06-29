import numpy as np

from anubis_logger import logger
from benchmarkers.direct_benchmarker import DirectBenchmarker


class NlpGenerativeBenchmarker(DirectBenchmarker):
    def __init__(self, run_config, data_loader, backends, batch_size, metrics_analysis=None):
        super(NlpGenerativeBenchmarker, self).__init__(run_config, data_loader, backends, batch_size, metrics_analysis)

    def run(self):
        res_benchmark = super().run()

        query_predict_times = []
        for bk in self._backends:
            query_predict_times.extend(bk.predict_times)

        token_predict_times_from_token_recoder = []
        for bk in self._backends:
            token_predict_times_from_token_recoder.extend(bk.token_predict_times)

        prompt_phase_predict_times = []
        token_phase_predict_times_from_token_recoder = []
        for t_times in token_predict_times_from_token_recoder:
            num_new_tokens = len(t_times)
            if num_new_tokens > 0:
                prompt_phase_predict_times.append(t_times[0])

            if num_new_tokens > 1:
                token_phase_predict_times_from_token_recoder.extend(t_times[1:])

        if self._run_config.token_metrics:
            self.__get_percentile_metrics_ms(res_benchmark, prompt_phase_predict_times, 'prompt_phase_latency_')

            token_phase_predict_times = []
            if len(query_predict_times) == len(prompt_phase_predict_times):
                for i in range(len(query_predict_times)):
                    token_phase_predict_times.append(query_predict_times[i] - prompt_phase_predict_times[i])
            self.__get_percentile_metrics_ms(res_benchmark, token_phase_predict_times, 'token_phase_latency_')

            if self._run_config.verbose:
                logger.info(f"query_predict_times\n{query_predict_times}")
                logger.info(f"prompt_phase_predict_times\n{prompt_phase_predict_times}")
                logger.info(f"token_phase_predict_times\n{token_phase_predict_times}")
                logger.info(f"token_predict_times_from_token_recoder\n{token_predict_times_from_token_recoder}")
                logger.info(f"token_phase_predict_times_from_token_recoder\n{token_phase_predict_times_from_token_recoder}")

            if token_phase_predict_times_from_token_recoder:
                self.__get_percentile_metrics_ms(res_benchmark, token_phase_predict_times_from_token_recoder, 'r_token_phase_latency_')
                res_benchmark["r_token/s"] = 1 / np.mean(token_phase_predict_times_from_token_recoder)

            res_benchmark["token/s"] = (self._run_config.max_new_tokens - 1) / np.mean(token_phase_predict_times) if token_phase_predict_times else 0

        return res_benchmark

    def __get_percentile_metrics_ms(self, res_benchmark, times, prefix):
        times_ms = [t * 1000 for t in times]
        self._get_percentile_metrics(res_benchmark, times_ms, prefix)
