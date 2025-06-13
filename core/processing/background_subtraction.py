import numpy as np
from .base import Processor

class BackgroundSubtraction(Processor):
    def __init__(self, region=(0,10)):
        """
        Usage:
        This processor subtracts the mean of a specified region from each scan in the data.
        Inputs:
        - region: a tuple (start_idx, end_idx) specifying the indices of the region to average for baseline subtraction.
        Output:
        - data: 2D numpy array where the mean of the specified region is subtracted from each color plot file.
        """
        self.region = region

    def process(self, data):
        start, end = self.region
        # compute mean CV over that region (axis=1 is voltage sweep)
        baseline = np.mean(data[:, start:end], axis=1, keepdims=True)
        #print(f"Baseline shape: {baseline.shape}, Data shape: {data.shape}")
        # subtract from every scan
        if np.array_equal(data,(data - baseline)):
            print("No change in data after background subtraction.")
        return data - baseline