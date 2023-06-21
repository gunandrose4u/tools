import numpy as np

from anubis_logger import logger
from benchmarkers.direct_benchmarker import DirectBenchmarker


class NlpGenerativeBenchmarker(DirectBenchmarker):
    def __init__(self, run_config, data_loader, backends, batch_size, metrics_analysis=None):
        super(NlpGenerativeBenchmarker, self).__init__(run_config, data_loader, backends, batch_size, metrics_analysis)

    def run(self):
        res_benchmark = super().run()

        prompt_predict_times = []
        for bk in self._backends:
            prompt_predict_times.extend(bk.predict_times)

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

        if self._run_config.token_metrics:
            prompt_predict_times_exclude_1st_token = []
            for i in range(len(prompt_predict_times)):
                prompt_predict_times_exclude_1st_token.append(prompt_predict_times[i] - first_token_predict_times[i])
            self._get_percentile_metrics(res_benchmark, prompt_predict_times_exclude_1st_token, 'prompt_latency_exclude_1st_token_')
            
            if not self._run_config.verbose:
                logger.info(f"prompt_predict_times\n{prompt_predict_times}")
                logger.info(f"token_predict_times\n{token_predict_times}")
                logger.info(f"first_token_predict_times\n{first_token_predict_times}")
                logger.info(f"prompt_predict_times_exclude_1st_token\n{prompt_predict_times_exclude_1st_token}")
                logger.info(f"non_first_token_predict_times\n{non_first_token_predict_times}")

            self._get_percentile_metrics(res_benchmark, first_token_predict_times, '1st_token_latency_')
            if non_first_token_predict_times:
                self._get_percentile_metrics(res_benchmark, non_first_token_predict_times, 'non_1st_token_latency_')
                res_benchmark["p_token/s"] = 1 / np.mean(non_first_token_predict_times)
            
            res_benchmark["token/s"] = (self._run_config.max_new_tokens - 1) / np.mean(prompt_predict_times_exclude_1st_token)

            
        return res_benchmark
