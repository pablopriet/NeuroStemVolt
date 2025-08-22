import numpy as np
from .base import Processor

class BackgroundSubtraction(Processor):
    """
    Subtracts the background signal based on a specified time region.

    This processor computes the mean signal over a given time window and subtracts it
    from every scan (column) in the data matrix. This is typically used to remove
    baseline current from color plot data.

    Args:
        region (tuple): Tuple of (start_time, end_time) in seconds indicating the
                        time range to average for background subtraction.
    """
    def __init__(self, region=(0,10)):
        self.region = region

    def process(self, data, context):
        """
        Apply background subtraction to FSCV color plot data.

        Args:
            data (np.ndarray): 2D array (voltage steps × time points).
            context (dict): Dictionary containing metadata. Must include key
                            "acquisition_frequency" (in Hz).

        Returns:
            np.ndarray: Background-subtracted 2D data array.
        """
        acq_freq = context["acquisition_frequency"]
        start, end = self.region
        start = start * acq_freq
        end = end * acq_freq

        print(start, end)
        # compute mean CV over that region (axis=1 is voltage sweep)
        baseline = np.mean(data[:, start:end], axis=1, keepdims=True)
        #print(f"Baseline shape: {baseline.shape}, Data shape: {data.shape}")
        # subtract from every scan
        if context is not None:
            context['background_subtraction_region'] = self.region
        if np.array_equal(data,(data - baseline)):
            print("No change in data after background subtraction.")
        return data - baseline