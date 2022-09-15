import os
import numpy as np
import random

from abc import abstractmethod, ABC
from anubis_logger import logger

FIXED_RANDOM_SEED = 2022

class Dataset(ABC):
    def __init__(self):
        self._data_x_inmemory = {}
        np.random.seed(FIXED_RANDOM_SEED)

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
        if isinstance(self._data_x_inmemory[id_list[0]], dict):
            input_names = self._data_x_inmemory[id_list[0]].keys()
            feed = {}
            for name in input_names:
                feed[name] = np.array([np.squeeze(self._data_x_inmemory[id][name], axis=0) for id in id_list])
            return feed
        else:
            return np.array([np.squeeze(self._data_x_inmemory[id], axis=0) for id in id_list])


class FileDataset(Dataset):
    def __init__(self, file_data_loader):
        super(FileDataset, self).__init__()
        self._file_data_loader = file_data_loader

    def get_all_items_count(self):
        return self._file_data_loader.loaded_count

    def load_query_samples(self, sample_list):
        self._data_x_inmemory = {}
        for sample in sample_list:
            self._data_x_inmemory[sample] = \
                self._file_data_loader.get_item_loc(sample % self._file_data_loader.loaded_count)

    def unload_query_samples(self, sample_list):
        self._data_x_inmemory = {}