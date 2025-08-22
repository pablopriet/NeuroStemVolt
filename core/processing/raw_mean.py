from .base import Processor
import numpy as np

class RawMean(Processor):
    """
    Apply a simple moving average (smoothing) across time.

    This performs 1D convolution using a flat kernel for each voltage step.

    Args:
        window_size (int): Width of the smoothing window.
    """
    def __init__(self, window_size=5):
        self.window_size = window_size

    def process(self, data):
        # Apply simple moving average across time axis (axis=0)
        # Note we are applying wrapping to handle edge cases (meaning our resulting data will have the same shape as input data)
        kernel = np.ones(self.window_size) / self.window_size
        smoothed_data = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode='same'),
            axis=0,
            arr=data
        ) 
        return smoothed_data
