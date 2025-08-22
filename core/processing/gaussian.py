import numpy as np
from .base import Processor

class GaussianSmoothing2D(Processor):
    def __init__(self, window=5, repeats=3, pad_mode='reflect'):
        """
        Approximates Gaussian smoothing using repeated box filtering.         

        Args:
            window (int): Width of box filter.
            repeats (int): Number of times to apply box filter along each axis.
            pad_mode (str): Padding strategy for edges.

        The implementation is inspired by the paper Novel, User-Friendly Experimental and Analysis Strategies for Fast Voltammetry: 1. The Analysis Kid for FSCV by Mena et al. (2021) 
        https://doi.org/10.1021/acs.analchem.1c01258
        """
        self.window   = window
        self.repeats  = repeats
        self.pad_mode = pad_mode

    def _box1d(self, arr, axis):
        # Pad so that 'valid' mode returns the same shape as input
        pad = [(0,0)] * arr.ndim
        pad[axis] = (self.window//2, self.window//2)
        padded = np.pad(arr, pad, mode=self.pad_mode)

        # Cumulative sum trick for O(1) per output:
        cs = np.cumsum(padded, axis=axis)

        # Build slices to compute windowed sums
        sl_end   = [slice(None)] * arr.ndim
        sl_start = [slice(None)] * arr.ndim
        sl_end[axis]   = slice(self.window, None)
        sl_start[axis] = slice(None, -self.window)

        summed = cs[tuple(sl_end)] - cs[tuple(sl_start)]
        return summed / self.window

    def process(self, data):
        """
        Apply 2D Gaussian-like smoothing via repeated box filters.

        Args:
            data (np.ndarray): 2D array to smooth.

        Returns:
            np.ndarray: Smoothed 2D array.
        """
        out = data.copy()

        # Horizontal (time) passes
        for _ in range(self.repeats):
            out = self._box1d(out, axis=1)

        # Vertical (voltage) passes
        for _ in range(self.repeats):
            out = self._box1d(out, axis=0)

        return out
