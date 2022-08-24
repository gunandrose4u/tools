
import os
import sys
import torch
from anubis_logger import logger
from backend import Backend

CUSTOMIZE_MODEL_LOADER_FILE_NAME = "model_loader"

class BackendPytorchNative(Backend):
    def __init__(self, kwargs):
        super(BackendPytorchNative, self).__init__(kwargs)
        self._model = None
        self._model_loader = None
        self._device = torch.device("cuda:0") if torch.cuda.is_available() and self._use_gpu else torch.device('cpu')
        if self._num_threads:
            torch.set_num_threads(int(self._num_threads))
        torch.set_grad_enabled(False)
        logger.info(f"pytorch is using device {self._device}")

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native"

    def load(self, model_path):
        model_folder = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        if os.path.exists(os.path.join(model_folder, f"{CUSTOMIZE_MODEL_LOADER_FILE_NAME}.py")):
            sys.path.append(model_folder)
            model_loader_module = __import__(CUSTOMIZE_MODEL_LOADER_FILE_NAME, fromlist=['ModelLoader'])
            self._model_loader = model_loader_module.ModelLoader()

        if self._model_loader:
            self._model = self._model_loader.load(model_path, self._device)
        else:
            try:
                self._model = torch.load(model_path)
            except:
                self._model = torch.load(model_path, map_location=self._device)

        self._model.eval()
        return self

    def predict(self, feed):
        if self._model_loader and hasattr(self._model_loader, "predict"):
            output = self._model_loader.predict(feed, self._device)
        else:
            # FIXME - this is not totally correct
            key = [key for key in feed.keys()][0]
            tmp_input = torch.from_numpy(feed[key]).to(self._device)
            output = self._model(tmp_input)

        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()

        return output
