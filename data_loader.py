import os
import numpy as np
import random

from abc import abstractmethod, ABC
from anubis_logger import logger

ACCEPT_FILE_TYPE = ".npy"

class DataLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_item(self):
        raise NotImplementedError()

class FileDataLoader(DataLoader):
    def __init__(self, data_folder, required_shape=None, required_dtype=None):
        super().__init__()
        self._data_folder = data_folder if os.path.isdir(data_folder) else os.path.dirname(data_folder)
        self._required_shape = required_shape
        self._required_dtype = required_dtype

        logger.info(f"Load data from {self._data_folder}")
        data_files = [os.path.join(self._data_folder, f) for f in os.listdir(self._data_folder) if f.endswith(ACCEPT_FILE_TYPE)]
        self._datas = []
        for f in data_files:
            input_feeds = np.load(f, allow_pickle=True)
            if required_shape and input_feeds.shape != required_shape:
                continue

            if required_dtype and input_feeds.dtype != required_dtype:
                continue

            self._datas.extend(input_feeds)
        logger.info(f"Load data from {self._data_folder} finished, items {len(self._datas)} loaded")

    def get_item(self, batch_size=1):
        if batch_size == 1:
            id = random.randrange(len(self._datas))
            return self._datas[id]
        elif batch_size > 1:
            id_list = [random.randrange(len(self._datas)) for i in range(batch_size)]
            if isinstance(self._datas[id_list[0]], dict):
                input_names = self._datas[id_list[0]].keys()
                feed = {}
                for name in input_names:
                    feed[name] = np.array([np.squeeze(self._datas[id][name], axis=0) for id in id_list])
                return feed
            else:
                return np.array([np.squeeze(self._datas[id], axis=0) for id in id_list])
        else:
            raise Exception("Batch size must larger than 0")
