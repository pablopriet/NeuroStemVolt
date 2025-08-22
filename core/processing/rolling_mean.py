from .base import Processor
import numpy as np

class RollingMean(Processor):
    """
    Smooth data with a rolling mean filter along the time axis.

    Args:
        window_size (int): Number of samples in moving window.
    """
    def __init__(self, window_size=5):
        self.window_size = window_size

    def process(self, data):
        """
        Apply a rolling mean (moving average) along the time dimension.

        Args:
            data (np.ndarray): 2D FSCV array.

        Returns:
            np.ndarray: Smoothed data array.
        """
        # Apply simple moving average across time axis (axis=0)
        # Note we are applying wrapping to handle edge cases (meaning our resulting data will have the same shape as input data)
        kernel = np.ones(self.window_size) / self.window_size
        smoothed_data = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, 'same'),
            axis=0,
            arr=data
        )
        return smoothed_data 
        
