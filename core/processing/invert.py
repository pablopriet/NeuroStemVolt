from .base import Processor

class InvertData(Processor):
    def process(self, data, context=None):
        """
        Invert the sign of the data.

        Args:
            data (np.ndarray): 2D FSCV array (voltage × time).
            context (dict): Optional metadata (not used).

        Returns:
            np.ndarray: Inverted data.
        """
        data = -data
        return data
