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
        warning_msg = None

        # Handle invalid regions gracefully
        if start_idx >= end_idx:
            warning_msg = (f"Invalid background region ({self.region[0]}s to {self.region[1]}s). "
                          f"Using default first 10 seconds.")
            print(f"WARNING: {warning_msg}")
            start_idx = 0
            end_idx = min(int(10 * acq_freq), n_time)
        
        if start_idx < 0:
            start_idx = 0
        if end_idx > n_time:
            end_idx = n_time
        
        # Ensure we have at least 2 scans
        if end_idx - start_idx < 2:
            warning_msg = f"Background region too narrow. Using first {min(10, n_time)} time points."
            print(f"WARNING: {warning_msg}")
            start_idx = 0
            end_idx = min(10, n_time)
        
        # Store warning in context for UI display
        if warning_msg:
            if 'processing_warnings' not in context:
                context['processing_warnings'] = []
            context['processing_warnings'].append(f"Background Subtraction: {warning_msg}")

        # Mean CV across the background time window
        #print(data.shape)
        baseline = np.mean(data[start_idx:end_idx, :], axis=0, keepdims=True)
        #print("Baseline Shape",baseline.shape)
        result = data - baseline

        # Store metadata
        context['background_subtraction_region'] = self.region
        context['background_subtraction_indices'] = (start_idx, end_idx)

        # Diagnostic: check if baseline has meaningful values
        baseline_range = np.max(baseline) - np.min(baseline)
        print(f"Background subtraction: region={self.region}s, indices=[{start_idx}:{end_idx}]")
        print(f"Baseline shape: {baseline.shape}, range: {baseline_range:.6f}")

        return result