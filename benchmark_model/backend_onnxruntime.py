import os
import onnxruntime as rt

from backend import Backend
from anubis_logger import logger


EXECUTION_MODE = {"ORT_SEQUENTIAL": rt.ExecutionMode.ORT_SEQUENTIAL, "ORT_PARALLEL": rt.ExecutionMode.ORT_PARALLEL}
GRAPH_OPTIMIZATION_LEVEL = {"ORT_DISABLE_ALL": rt.GraphOptimizationLevel.ORT_DISABLE_ALL,
                            "ORT_ENABLE_BASIC": rt.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                            "ORT_ENABLE_EXTENDED": rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                            "ORT_ENABLE_ALL": rt.GraphOptimizationLevel.ORT_ENABLE_ALL,
                            }


class BackendOnnxruntime(Backend):
    def __init__(self, kwargs):
        super(BackendOnnxruntime, self).__init__(kwargs)
        self._device_id = kwargs.get("device_id", None)
        self._user_specified_ep = os.environ.get('ORT_EP')

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "onnxruntime"

    def load(self, model_path):
        """Load model and find input/outputs from the model file."""

        providers, provider_opt = self._get_provider()
        logger.info(f"Used provider = {providers}, {provider_opt}")

        opt = rt.SessionOptions()
        opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._num_threads:
            opt.intra_op_num_threads = int(self._num_threads)

        if self._optioins:
            self._set_session_options(opt)

        self.sess = rt.InferenceSession(model_path, sess_options=opt, providers=providers, provider_options=provider_opt)

        self._inputs = [meta.name for meta in self.sess.get_inputs()]
        self._outputs = [meta.name for meta in self.sess.get_outputs()]
        inputs_info = { meta.name: meta.shape for meta in self.sess.get_inputs()}
        logger.info(f"model_inputs:{inputs_info}")
        return self

    def predict(self, feed):
        """Run the prediction."""
        return self.sess.run(self._outputs, feed)

    def _set_session_options(self, sess_opt):
        for k in self._optioins.keys():
            if k == "session_config_entry":
                self._set_session_config_entry(sess_opt, self._optioins[k])
                continue

            if hasattr(sess_opt, k):
                try:
                    if k == 'execution_mode':
                        self._optioins[k] = EXECUTION_MODE[k]
                    elif k == 'graph_optimization_level':
                        self._optioins[k] = GRAPH_OPTIMIZATION_LEVEL[k]

                    setattr(sess_opt, k, self._optioins[k])
                    logger.info(f"Set session option {k}={self._optioins[k]}")
                except:
                    logger.warning(f"Can not set session option {k}={self._optioins[k]}")
            else:
                logger.warning(f"Session option does not contain property {k}")

    def _set_session_config_entry(self, sess_opt, configs):
        if not configs:
            logger.warning(f"Session config entry is empty")
            return

        for k in configs.keys():
            try:
                sess_opt.add_session_config_entry(k, configs[k])
                logger.info(f"Set session option config entry {k}={configs[k]}")
            except:
                logger.warning(f"Can not set session option config entry {k}={configs[k]}")


    def _get_provider(self):
        support_providers = rt.get_available_providers()
        logger.info(f"support_providers = {support_providers}")

        res_provider_options = []
        res_providers = ["CPUExecutionProvider"]

        if self._user_specified_ep:
            logger.info(f"User set EP by env var to {self._user_specified_ep}")
            res_providers = [self._user_specified_ep]
        elif self._use_gpu:
            res_providers = ["ROCMExecutionProvider" if 'ROCMExecutionProvider' in support_providers else "CUDAExecutionProvider"]

        if res_providers[0] == "OpenVINOExecutionProvider":
            device = os.environ.get('ORT_OPENVINO_EP_DEVICE')
            if not device:
                device = "CPU_FP32"
            res_provider_options = [{'device_type' : device}]
            logger.info("Device type selected is: " + device + " using the OpenVINO Execution Provider")
        elif res_providers[0] == "CUDAExecutionProvider" or res_providers[0] == "ROCMExecutionProvider":
            if self._device_id is not None:
                res_providers = [
                    (res_providers[0], {
                        'device_id': self._device_id,
                    })
                ]

        return res_providers, res_provider_options if res_provider_options else None
