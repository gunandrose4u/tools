import os
import onnxruntime as rt

from backend import Backend
from anubis_logger import logger

class BackendOnnxruntime(Backend):
    def __init__(self, kwargs):
        super(BackendOnnxruntime, self).__init__(kwargs)
        self._device_id = kwargs.get("device_id", None)

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "onnxruntime"

    def load(self, model_path):
        """Load model and find input/outputs from the model file."""

        providers = self._get_provider()
        logger.info(f"Used provider = {providers}")

        opt = rt.SessionOptions()
        opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._num_threads:
            opt.intra_op_num_threads = int(self._num_threads)

        self.sess = rt.InferenceSession(model_path, sess_options=opt, providers=providers)

        self._inputs = [meta.name for meta in self.sess.get_inputs()]
        self._outputs = [meta.name for meta in self.sess.get_outputs()]
        inputs_info = { meta.name: meta.shape for meta in self.sess.get_inputs()}
        logger.info(f"model_inputs:{inputs_info}")
        return self

    def predict(self, feed):
        """Run the prediction."""
        return self.sess.run(self._outputs, feed)

    def _get_provider(self):
        support_providers = rt.get_available_providers()
        logger.info(f"support_providers = {support_providers}")
        providers = ['CPUExecutionProvider']

        if self._use_gpu:
            logger.info(f"self._device_id = {self._device_id}")
            if 'ROCMExecutionProvider' in support_providers:
                if self._device_id is not None:
                    providers = [
                        ('ROCMExecutionProvider', {
                            'device_id': self._device_id,
                        })
                    ]
                else:
                    providers = ['ROCMExecutionProvider']
            else:
                providers = ['CUDAExecutionProvider']

        return providers