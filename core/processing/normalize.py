from .base import Processor
from .base import Processor
import numpy as np
from scipy.signal import find_peaks

class Normalize(Processor):
    def __init__(self, peak_position=257):
        self.peak_position = peak_position
    def process(self, data,context=None):
        """
        Usage:
        This processor normalizes the input data by dividing each scan by the maximum value of that scan.
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