import sys
import pathlib

from abc import abstractmethod, ABC

class DataLoader(ABC):
    def __init__(self, run_config):
        self._loaded_data_x = []
        self._run_config = run_config

    @abstractmethod
    def get_batch_items(self, batch_size=1):
        raise NotImplementedError()

    def get_item_loc(self, loc):
        return self._loaded_data_x[loc]

    @property
    def loaded_count(self):
        return len(self._loaded_data_x)

    def post_process(self, results):
        pass


class DataLoaderFactory():
    def __init__(self):
        curdir = pathlib.Path(__file__).parent.resolve()
        sys.path.append(str(curdir))

    def get_data_loader(self, run_config):
        loader_module = __import__(run_config.dataloader, fromlist=['BenchmarkDataLoader'])
        return loader_module.BenchmarkDataLoader(run_config)
