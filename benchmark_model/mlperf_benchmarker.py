# """""""""""""""""""""""""""""""""""""""""""
# Refactor from mlperf-bench project
# """""""""""""""""""""""""""""""""""""""""""


import os
import re
import datetime
import array
import numpy as np
import threading
import mlperf_loadgen as lg

from abc import abstractmethod, ABC
from anubis_logger import logger
from benchmarker import Benchmarker
from utilities import load_json_from_file
from dataset import FileDataset
from queue import Queue
from time import perf_counter

class ScenarioProcessor(ABC):
    def __init__(self, process_fn, batch_size, dataset):
        self._process_fn = process_fn
        self._batch_size = batch_size
        self._ds = dataset

    def dispatch(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self._batch_size:
            feed = self._ds.make_batch(idx)
            self.process_batch_query((query_id, idx, feed))
        else:
            bs = self._batch_size
            for i in range(0, len(idx), bs):
                feed = self._ds.make_batch(idx[i:i + bs])
                self.process_batch_query((query_id[i:i + bs], idx[i:i + bs], feed))

    @abstractmethod
    def process_batch_query(self, batch_query):
        raise NotImplementedError()


class SingleStreamProcessor(ScenarioProcessor):
    def __init__(self, process_fn, batch_size, dataset, threads=1, backends_count=1):
        super(SingleStreamProcessor, self).__init__(process_fn, 1, dataset)
        self._backends_count = backends_count
        self._backend_selector = 0

    def process_batch_query(self, batch_query):
        self._process_fn(batch_query, self._backend_selector % self._backends_count)
        self._backend_selector += 1


class OfflineProcessor(ScenarioProcessor):
    def __init__(self, process_fn, batch_size, dataset, threads=2, backends_count=1):
        super(OfflineProcessor, self).__init__(process_fn, batch_size, dataset)
        self._batch_query_dispatch_queue = Queue(maxsize=threads * 4)
        self._workers = []

        for bk_id in range(backends_count):
            for _ in range(threads):
                worker = threading.Thread(target=self._handle_tasks, args=(self._batch_query_dispatch_queue, bk_id, ))
                worker.daemon = True
                self._workers.append(worker)
                worker.start()

    def _handle_tasks(self, query_queue, bk_id):
        """Worker thread."""
        while True:
            batch_query = query_queue.get()
            query_queue.task_done()
            if not batch_query:
                # None in the queue indicates the parent want us to exit
                break
            self._process_fn(batch_query, bk_id)

    def process_batch_query(self, batch_query):
        self._batch_query_dispatch_queue.put(batch_query)


SCENARIO_MAPPING = {
    "SingleStream": [SingleStreamProcessor, lg.TestScenario.SingleStream],
    "Offline": [OfflineProcessor, lg.TestScenario.Offline]
}

class MlPerfBenchmarker(Benchmarker):
    def __init__(self, config, data_loader, backends, batch_size, metrics_analysis=None):
        super(MlPerfBenchmarker, self).__init__(config, data_loader, backends, batch_size, metrics_analysis)
        self._mlperf_config = load_json_from_file(self._config.mlperf_config)
        self._ds = FileDataset(data_loader)
        self._post_process = None
        self._log_output_settings = lg.LogOutputSettings()
        self._log_settings = lg.LogSettings()
        self._test_settings = lg.TestSettings()

        if self._config.mlperf_scenario not in SCENARIO_MAPPING.keys():
            raise Exception(f"Only support scenarios as {SCENARIO_MAPPING.keys()}")

        self._init_mlperf_log_settings(self._mlperf_config["logSettings"])
        self._init_mlperf_test_settings(self._mlperf_config["testSettings"])

        self._scenario_processor = SCENARIO_MAPPING[self._config.mlperf_scenario][0](self._run_one_item, batch_size, self._ds, self._config.num_runner_threads, len(self._backends))
        self._sut = lg.ConstructSUT(self._issue_queries, self._flush_queries, self._process_latencies)
        self._qsl = lg.ConstructQSL(self._i_total_samples, max(self._ds.get_all_items_count(), 500), self._ds.load_query_samples, self._ds.unload_query_samples)

    def _init_mlperf_log_settings(self, log_settings):
        self._log_output_settings.outdir = log_settings["outputDir"] if log_settings["outputDir"] else "mlperf_results"
        self._log_output_settings.copy_summary_to_stdout = log_settings["copySummaryToStdout"]

        if not os.path.exists(self._log_output_settings.outdir):
            os.mkdir(self._log_output_settings.outdir)

        self._log_settings = lg.LogSettings()
        self._log_settings.enable_trace = log_settings["enableTrace"]
        self._log_settings.log_output = self._log_output_settings

    def _init_mlperf_test_settings(self, test_settings):
        self._test_settings.FromConfig("mlperf.conf", self._config.model, self._config.mlperf_scenario)
        if test_settings["fromFile"]:
            self._test_settings.FromConfig(test_settings["fromFile"], self._config.model, self._config.mlperf_scenario)

        self._test_settings.scenario = SCENARIO_MAPPING[self._config.mlperf_scenario][1]
        self._test_settings.mode = lg.TestMode.PerformanceOnly

        override_values = test_settings["overrideValues"]
        for k in override_values.keys():
            if hasattr(self._test_settings, k):
                try:
                    setattr(self._test_settings, k, override_values[k])
                    logger.info(f"Set test settings {k}={override_values[k]}")
                except:
                    logger.warning(f"Can not set test settings {k}={override_values[k]}")
            else:
                logger.warning(f"Test settings does not contain property {k}")

    def _issue_queries(self, query_samples):
        self._scenario_processor.dispatch(query_samples)

    def _flush_queries(self):
        pass

    def _process_latencies(self, latencies_ns):
        pass

    def _run_one_item(self, qitem, bk_id):
        # run the prediction
        processed_results = []
        query_id, content_id, feed = qitem
        results = self._backends[bk_id].predict_with_perf(feed)

        if self._post_process:
            processed_results = self._post_process(results, content_id)
        else:
            processed_results = [[]] * len(query_id)

        response_array_refs = []
        response = []
        for idx, qid in enumerate(query_id):
            response_array = array.array("B", np.array(processed_results[idx], np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)

    def _parse_mlperf_log(self):
        rt = {}
        is_valid = False
        fname = os.path.join(self._log_output_settings.outdir, "mlperf_log_summary.txt")
        with open(fname, "r") as f:
            for line in f:
                m = re.match(r"^Result\s+is\s*\:\s+VALID", line)
                if m:
                    is_valid = True
                m = re.match(r"^\s*([\w\s.\(\)\/]+)\s*\:\s*([\w\+\.]+).*", line)
                if m:
                    rt[m.group(1).strip()] = m.group(2).strip()

        def ns(n):
            return "{:.2f}".format(float(rt[n]) / 1000000.)

        def yes_no(n):
            if n.lower() in ["yes", "true", "valid"]:
                return 1
            return 0

        def metric(scenario):
            if scenario in ["Offline"]:
                return "Samples per second"
            if scenario in ["SingleStream"]:
                return "QPS w/o loadgen overhead"
            if scenario in ["MultiStream"]:
                return "Samples per query"
            if scenario in ["Server"]:
                return "Scheduled samples per second"
            return None

        res_benchmark = {}
        res_benchmark["filter"] = 0
        res_benchmark["mode"] = rt["Mode"]
        res_benchmark["scenario"] = self._config.mlperf_scenario
        res_benchmark["qps"] = rt.get(metric(self._config.mlperf_scenario))
        res_benchmark["mean"] = ns('Min latency (ns)')
        res_benchmark["min"] = ns('Min latency (ns)')
        res_benchmark["max"] = ns('Max latency (ns)')
        res_benchmark["50pt"] = ns('50.00 percentile latency (ns)')
        res_benchmark["90pt"] = ns('90.00 percentile latency (ns)')
        res_benchmark["95pt"] = ns('95.00 percentile latency (ns)')
        res_benchmark["99pt"] = ns('99.00 percentile latency (ns)')
        res_benchmark["99.9pt"] = ns('99.90 percentile latency (ns)')
        res_benchmark["valid"] = is_valid
        res_benchmark["perf_ok"] = yes_no(rt.get('Performance constraints satisfied', "YES"))
        res_benchmark["mindur_ok"] = yes_no(rt['Min duration satisfied'])
        res_benchmark["minqs_ok"] = yes_no(rt['Min queries satisfied'])

        return res_benchmark

    def run(self):
        logger.info(f"Start benchmarking as mlperf way, scenario is {self._config.mlperf_scenario}")
        t2_start = perf_counter()
        lg.StartTestWithLogSettings(self._sut, self._qsl, self._test_settings, self._log_settings)
        t2_end = perf_counter()
        lg.DestroyQSL(self._qsl)
        lg.DestroySUT(self._sut)
        logger.info(f"Benchmark finished")

        predict_times_in_ms = [ i * 1000 for i in self._backends[0].predict_times]

        res_benchmark = self._parse_mlperf_log()
        res_benchmark['time'] = datetime.datetime.now().strftime("%m/%d/%Y %H:%M")
        res_benchmark['model'] = self._config.model
        res_benchmark['batch_size'] = self._batch_size
        res_benchmark['warmup_times'] = self._config.warmup_times
        res_benchmark['use_gpu'] = self._config.use_gpu
        res_benchmark['var'] = np.std(predict_times_in_ms) / np.mean(predict_times_in_ms)
        res_benchmark['std'] = np.std(predict_times_in_ms)
        res_benchmark['duration'] = t2_end - t2_start

        return res_benchmark
