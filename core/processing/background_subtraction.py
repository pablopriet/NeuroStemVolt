import numpy as np
from .base import Processor

class BackgroundSubtraction(Processor):
    def __init__(self, region=(0,10)):
        """
        region: a (start_idx, end_idx) tuple of scan‐indices to average for baseline
        """
        self.region = region

    def process(self, data):
        start, end = self.region
        # compute mean CV over that region (axis=1 is voltage sweep)
        baseline = np.mean(data[:, start:end], axis=1, keepdims=True)
        # subtract from every scan
        if np.array_equal(data,(data - baseline)):
            print("No change in data after background subtraction.")
        return data - baseline