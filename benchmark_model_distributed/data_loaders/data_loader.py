import sys
import pathlib
import numpy as np

from abc import abstractmethod, ABC


class DataLoader(ABC):
    def __init__(self, run_config):
        self._loaded_data_x = []
        self._run_config = run_config
        np.random.seed(20230621)

    def get_batch_items(self, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size should be greater than 0")

        if batch_size == 1:
            return self.get_item_loc(np.random.randint(len(self._loaded_data_x)))
        else:
            batch_indices = [np.random.randint(len(self._loaded_data_x)) for _ in range(batch_size)]
            return self.make_batch(self._loaded_data_x, batch_indices)

    @abstractmethod
    def make_batch(self, data_array, selected_indeices):
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
        self._curdir = pathlib.Path(__file__).parent.resolve()
        sys.path.append(str(self._curdir))

    def get_data_loader(self, run_config):
        splited_dl_name = run_config.dataloader.split('.')
        if len(splited_dl_name) > 1:
            dl_module_path = str(self._curdir) +  "/".join(splited_dl_name[0:-1])
            sys.path.append(dl_module_path)
        loader_module = __import__(splited_dl_name[-1], fromlist=['BenchmarkDataLoader'])
        return loader_module.BenchmarkDataLoader(run_config)
