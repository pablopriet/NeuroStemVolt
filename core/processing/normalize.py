from .base import Processor
import numpy as np


class Normalize(Processor):
    """
    Normalize data by the peak amplitude found by FindAmplitude.

    Expects FindAmplitude to run BEFORE this processor and set 'peak_amplitude_values'
    in the context. The first file's peak value is stored as 'experiment_first_peak'
    and used to normalize all subsequent files in the experiment.

    Args:
        peak_position (int): Index in voltage axis (kept for API compatibility).
    """
    def __init__(self, peak_position=257):
        self.peak_position = peak_position

    def process(self, data, context=None):
        """
        Normalize data using the peak amplitude from FindAmplitude.

        The first file processed will have its peak amplitude stored in context
        as 'experiment_first_peak'. All subsequent files will be normalized 
        using that same baseline value.

        Args:
            data (np.ndarray): 2D FSCV array (voltage × time).
            context (dict, optional): Must contain 'peak_amplitude_values' from FindAmplitude.

        Returns:
            np.ndarray: Normalized data.
        """
        # Check if we already have the normalization factor from a previous file
        if context is not None and "experiment_first_peak" in context:
            norm_factor = context["experiment_first_peak"]
            print(f"Normalize: Using stored experiment_first_peak = {norm_factor:.6f}")
        elif context is not None and "peak_amplitude_values" in context:
            # First file: use the peak value found by FindAmplitude
            norm_factor = context["peak_amplitude_values"]
            print(f"Normalize: First file peak_amplitude_values = {norm_factor}")
            
            # Handle array vs scalar
            if isinstance(norm_factor, np.ndarray):
                norm_factor = float(norm_factor.flat[0])
                print(f"Normalize: Converted from array to scalar: {norm_factor:.6f}")
            
            # Validate the normalization factor
            if norm_factor == 0 or np.isnan(norm_factor) or np.abs(norm_factor) < 1e-10:
                print("Warning: Invalid normalization factor from FindAmplitude, defaulting to 1.0")
                norm_factor = 1.0
            
            # Store for subsequent files
            context["experiment_first_peak"] = norm_factor
            print(f"Normalization baseline set from first file: {norm_factor:.6f}")
        else:
            # Fallback if FindAmplitude didn't run or context is None
            print("Warning: No peak amplitude in context. Using max value as fallback.")
            fx = data[:, self.peak_position]
            norm_factor = np.max(np.abs(fx))
            if norm_factor == 0 or np.isnan(norm_factor):
                norm_factor = 1.0
            if context is not None:
                context["experiment_first_peak"] = norm_factor
        
        normalized_data = data / norm_factor
        return normalized_data