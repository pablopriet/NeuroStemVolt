import numpy as np
from .base import Processor


class BackgroundSubtraction(Processor):
    """
    Subtracts the background signal based on a specified time region.

    Computes the mean cyclic voltammogram over a given time window and subtracts
    it from every scan in the data matrix, revealing faradaic current changes.

    Args:
        region (tuple): (start_time, end_time) in seconds for the background window.
    """

    def __init__(self, region=(0, 10)):
        self.region = region

    def process(self, data, context):
        """
        Apply background subtraction to FSCV color plot data.

        Args:
            data (np.ndarray): 2D array (voltage_steps × time_points).
            context (dict): Must include "acquisition_frequency" (Hz).

        Returns:
            np.ndarray: Background-subtracted 2D data array.
        """
        acq_freq = context["acquisition_frequency"]
        start_idx = int(self.region[0] * acq_freq)
        end_idx = int(self.region[1] * acq_freq)

        n_voltage, n_time = data.shape

        # Validate indices
        if start_idx < 0 or end_idx > n_time or start_idx >= end_idx:
            raise ValueError(
                f"Invalid background region indices [{start_idx}:{end_idx}] "
                f"for data with {n_time} time points. "
                f"Region={self.region}s, acq_freq={acq_freq}Hz"
            )

        if end_idx - start_idx < 2:
            raise ValueError(
                f"Background region too narrow: only {end_idx - start_idx} scans. "
                f"Increase the time window."
            )

        # Mean CV across the background time window
        print(data.shape)
        baseline = np.mean(data[start_idx:end_idx, :], axis=0, keepdims=True)
        print("Baseline Shape",baseline.shape)
        result = data - baseline

        # Store metadata
        context['background_subtraction_region'] = self.region
        context['background_subtraction_indices'] = (start_idx, end_idx)

        # Diagnostic: check if baseline has meaningful values
        baseline_range = np.max(baseline) - np.min(baseline)
        print(f"Background subtraction: region={self.region}s, indices=[{start_idx}:{end_idx}]")
        print(f"Baseline shape: {baseline.shape}, range: {baseline_range:.6f}")

        return result