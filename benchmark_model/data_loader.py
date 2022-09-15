import os
import numpy as np
import random

from abc import abstractmethod, ABC
from anubis_logger import logger

ACCEPT_FILE_TYPE = ".npy"

class DataLoader(ABC):
    def __init__(self):
        self._loaded_data_x = []

    @abstractmethod
    def get_batch_items(self, batch_size=1):
        raise NotImplementedError()

    def get_item_loc(self, loc):
        return self._loaded_data_x[loc]

    @property
    def loaded_count(self):
        return len(self._loaded_data_x)


class FileDataLoader(DataLoader):
    def __init__(self, data_folder, required_shape=None, required_dtype=None):
        super().__init__()
        self._data_folder = data_folder if os.path.isdir(data_folder) else os.path.dirname(data_folder)
        self._required_shape = required_shape
        self._required_dtype = required_dtype

        logger.info(f"Load data from {self._data_folder}")
        data_files = [os.path.join(self._data_folder, f) for f in os.listdir(self._data_folder) if f.endswith(ACCEPT_FILE_TYPE)]
        self._loaded_data_x = []
        for f in data_files:
            input_feeds = np.load(f, allow_pickle=True)
            if required_shape and input_feeds.shape != required_shape:
                continue

            if required_dtype and input_feeds.dtype != required_dtype:
                continue

            self._loaded_data_x.extend(input_feeds)

        if len(self._loaded_data_x) == 0:
            raise Exception(f"No data loaded from {data_folder}")

        logger.debug(self._loaded_data_x[0])
        logger.info(f"Load data from {self._data_folder} finished, items {len(self._loaded_data_x)} loaded")

    def get_batch_items(self, batch_size=1):
        if batch_size == 1:
            id = random.randrange(len(self._loaded_data_x))
            return self._loaded_data_x[id]
        elif batch_size > 1:
            id_list = [random.randrange(len(self._loaded_data_x)) for i in range(batch_size)]
            return self._get_batch_items_by_ids(id_list)
        else:
            raise Exception("Batch size must larger than 0")

    def _get_batch_items_by_ids(self, id_list):
        if isinstance(self._loaded_data_x[id_list[0]], dict):
            input_names = self._loaded_data_x[id_list[0]].keys()
            feed = {}
            for name in input_names:
                feed[name] = np.array([np.squeeze(self._loaded_data_x[id][name], axis=0) for id in id_list])
            return feed
        else:
            return np.array([np.squeeze(self._loaded_data_x[id], axis=0) for id in id_list])


class DataLoaderFactory():
    def __init__(self):
        pass

    def get_data_loader(self, config, data_path):
        return FileDataLoader(data_path)
