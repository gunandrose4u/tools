import time
import numpy as np

class Backend():
    def __init__(self, kwargs):
        self.predict_times = []
        self._num_threads = kwargs.get("num_threads")
        self._use_gpu = kwargs.get("use_gpu")

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


def get_backend(backend, kwargs):
    if backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime
        backend = BackendOnnxruntime(kwargs)
    elif backend == "torch":
        from backend_pytorch_native import BackendPytorchNative
        backend = BackendPytorchNative(kwargs)
    else:
        raise ValueError("unknown backend: " + backend)
    return backend


VALID_BACKENDS = ["onnxruntime", "pytorch-native"]
