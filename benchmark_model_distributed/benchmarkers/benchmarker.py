from abc import abstractmethod, ABC
from anubis_logger import logger


class Benchmarker(ABC):
    def __init__(self, run_config, data_loader, backends, batch_size, metrics_analysis):
        self._data_loader = data_loader
        self._backends = backends
        self._metrics_analysis = metrics_analysis
        self._run_config = run_config
        self._batch_size = batch_size

        self._test_times = self._run_config.test_times

    def warmup(self):
        logger.info(f"Warmup model for {self._run_config.warmup_times} times")
        for i in range(self._run_config.warmup_times):
            for bk in self._backends:
                bk.predict(self._data_loader.get_batch_items(self._batch_size))
        logger.info(f"Warmup finished")

    @abstractmethod
    def run(self):
        raise NotImplementedError()
