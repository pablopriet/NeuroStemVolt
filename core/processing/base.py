from abc import ABC, abstractmethod

class Processor(ABC):
    @abstractmethod
    def process(self, data, context=None):
        pass