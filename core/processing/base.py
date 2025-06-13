from abc import ABC, abstractmethod

class Processor(ABC):
    """
    Base class for all processors in the pipeline.
    Usage:
        This class defines the base class for all processors that will be used in the processing pipeline.
    """
    @abstractmethod
    def process(self, data, context=None):
        pass