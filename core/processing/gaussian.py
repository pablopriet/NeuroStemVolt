import numpy as np
from .base import Processor

class GaussianSmoothing2D(Processor):
    def __init__(self, window=5, repeats=3, pad_mode='reflect'):
        """
        window    : int, width of the uniform (box) kernel
        repeats   : int, number of times to apply the box filter per axis (3≈Gaussian)
        pad_mode  : str, how to extend data at the edges ('reflect' is usually best)
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
        data: 2D numpy array shape (n_voltage_bins, n_timepoints)
        returns: blurred 2D array, same shape
        """
        out = data.copy()

        # Horizontal (time) passes
        for _ in range(self.repeats):
            out = self._box1d(out, axis=1)

        # Vertical (voltage) passes
        for _ in range(self.repeats):
            out = self._box1d(out, axis=0)

        return out
