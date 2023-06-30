from abc import abstractmethod, ABC

from benchmarkers.mlperf.const import PERFORMANCEONLY_MODE, SINGLESTREAM

class MlPerfDataset(ABC):
    def __init__(self):
        self._data_x_inmemory = {}
        self._squeeze = True

    @abstractmethod
    def get_all_items_count(self):
        raise NotImplementedError()

    @abstractmethod
    def load_query_samples(self, sample_list):
        raise NotImplementedError()

    @abstractmethod
    def unload_query_samples(self, sample_list):
        raise NotImplementedError()

    @property
    def shapes(self):
        raise NotImplementedError()

    def make_batch(self, id_list):
        raise NotImplementedError()

# Add following detection in the future if we need make sure the data loader returns a torch tensor
# if not torch.is_tensor(pt_tensor):
#     raise ValueError("The data loader should return a torch tensor")
# if pt_tensor.get_device() != torch.device('cpu'):
#     pt_tensor = pt_tensor.cpu()

class DataLoaderToMlPerfDataset(MlPerfDataset):
    def __init__(self, data_loader, mlperf_scenario, mlperf_mode, batch_size_item=1):
        super(DataLoaderToMlPerfDataset, self).__init__()
        self._data_loader = data_loader
        self._mlperf_scenario = mlperf_scenario
        self._mlperf_mode = mlperf_mode

        if mlperf_mode == PERFORMANCEONLY_MODE and mlperf_scenario == SINGLESTREAM:
            assert batch_size_item > 0, "batch_size_item should be greater than 0"
            self._batch_size_item = batch_size_item
        else:
            self._batch_size_item = 1

    def get_all_items_count(self):
        return self._data_loader.loaded_count

    def load_query_samples(self, sample_list):
        self._data_x_inmemory = {}
        for sample in sample_list:
            data_tensor = self._data_loader.get_batch_items(self._batch_size_item)
            self._data_x_inmemory[sample] = data_tensor

    def unload_query_samples(self, sample_list):
        self._data_x_inmemory = {}

    def make_batch(self, id_list):
        if len(id_list) == 1:
            return self._data_x_inmemory[id_list[0]]
        else:
            return self._data_loader.make_batch(self._data_x_inmemory, id_list)