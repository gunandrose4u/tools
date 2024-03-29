import sys
import pathlib

from time import perf_counter

class Backend():
    def __init__(self, run_config):
        self._run_config = run_config
        self.predict_times = []
        self.start_predict_time = None
        self.end_predict_time = None

    def model_info(self):
        raise NotImplementedError("Backend:model_info")

    def version(self):
        raise NotImplementedError("Backend:version")

    def name(self):
        raise NotImplementedError("Backend:name")

    def load_model(self):
        raise NotImplementedError("Backend:load_model")

    def predict(self, inputs):
        raise NotImplementedError("Backend:predict")

    def predict_with_perf(self, inputs):
        self.start_predict_time = perf_counter()
        res = self.predict(inputs)
        self.end_predict_time = perf_counter()
        self.predict_times.append((self.end_predict_time-self.start_predict_time))
        return res

    def clear_perf_details(self):
        self.predict_times.clear()

class NlpGenerativeBackend(Backend):
    def __init__(self, run_config):
        super().__init__(run_config)
        self.token_predict_times = []

class BackendFactory():
    def __init__(self):
        curdir = pathlib.Path(__file__).parent.resolve()
        sys.path.append(str(curdir))

    def get_backend(self, run_config):
        backend_module = __import__(run_config.backend_name, fromlist=['BenchmarkBackend'])
        return backend_module.BenchmarkBackend(run_config)
