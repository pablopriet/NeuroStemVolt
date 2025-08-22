from .base import Processor
from .base import Processor
import numpy as np
from scipy.signal import find_peaks

class Normalize(Processor):
    """
    Normalize data by the mean peak value in the I-T profile.

    If a peak value is already present in the context under "experiment_first_peak",
    use it for normalization. Otherwise, compute and store it from the current trace.

    Args:
        peak_position (int): Index in voltage axis used for I-T profile.
    """
    def __init__(self, peak_position=257):
        self.peak_position = peak_position
    def process(self, data,context=None):
        """
        Normalize each scan in the data using the first peak amplitude.

        Args:
            data (np.ndarray): 2D FSCV array (voltage × time).
            context (dict, optional): Stores or uses 'experiment_first_peak'.

        Returns:
            np.ndarray: Normalized data.
        """
        if context is not None:
            if "experiment_first_peak" in context:
                # Normalize each scan by the first peak value
                normalized_data = data / context["experiment_first_peak"]
            else:
                fx = data[:, self.peak_position]
                peaks, _ = find_peaks(fx, prominence=0.2, distance=10, height=0.1)
                peak_values = fx[peaks]
                context["experiment_first_peak"] = np.mean(peak_values, axis=0)
                normalized_data = data / context["experiment_first_peak"]
        return normalized_data