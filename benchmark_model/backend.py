import time
import numpy as np

class Backend():
    def __init__(self, kwargs):
        self.predict_times = []
        self._num_threads = kwargs.get("num_threads")
        self._use_gpu = kwargs.get("use_gpu", False)
        self._optioins = kwargs.get("options", False)
        self.loaded_model = None

    def version(self):
        raise NotImplementedError("Backend:version")

    def name(self):
        raise NotImplementedError("Backend:name")

    def load(self, model_path):
        raise NotImplementedError("Backend:load")

    def predict(self, feed):
        raise NotImplementedError("Backend:predict")

    def predict_with_perf(self, feed):
        start = time.perf_counter()
        res = self.predict(feed)
        end = time.perf_counter()
        self.predict_times.append((end-start))
        return res

    def clear_perf_details(self):
        self.predict_times.clear()


class BackendFactory():
    VALID_BACKENDS = ["onnxruntime", "pytorch-native"]

    def __init__(self):
        pass

    def get_backend(self, config, visable_device=0):
        bk_opts = self._build_backend_options(config, visable_device)
        if config.framework == "onnxruntime":
            from backend_onnxruntime import BackendOnnxruntime
            backend = BackendOnnxruntime(bk_opts)
        elif config.framework == "torch":
            from backend_pytorch_native import BackendPytorchNative
            backend = BackendPytorchNative(bk_opts)
        else:
            raise ValueError("unknown backend: " + config.framework)
        return backend

    def _build_backend_options(self, config, visable_device):
        backend_opts = {}
        if config.num_threads > 0:
            backend_opts['num_threads'] = config.num_threads

        if config.use_gpu:
            backend_opts['use_gpu'] = config.use_gpu

            if visable_device >= 0:
                backend_opts['device_id'] = visable_device

        if config.ort_io_binding:
            backend_opts['ort_io_binding'] = config.ort_io_binding

        return backend_opts
