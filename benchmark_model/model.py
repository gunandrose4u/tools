from abc import abstractmethod, ABC



class Model(ABC):
    def __init__(self):
        self._backend = None


    @property
    def inputs(self):
        raise NotImplementedError()

    @property
    def outputs(self):
        raise NotImplementedError()

    @abstractmethod
    def benchmark(self):
        raise NotImplementedError()
