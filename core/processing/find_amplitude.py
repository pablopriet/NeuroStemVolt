from .base import Processor
import numpy as np
from scipy.signal import find_peaks

class FindAmplitude(Processor):
    def __init__(self, peak_position=257):
        self.peak_position = peak_position

    def process(self, data, context=None):
        """
        Usage:
        Finds the dominant local maxima of the input data and updates the context with their positions and values.
        """
        fx = data[:, self.peak_position]
        peaks, _ = find_peaks(fx, prominence=0.2, distance=10, height=0.1)

        peak_positions = peaks
        peak_values = fx[peaks]

        # Update context with peak information
        if context is not None:
            context['peak_amplitude_positions'] = peak_positions
            context['peak_amplitude_values'] = peak_values
            if "experiment_first_peak" not in context:
                context["experiment_first_peak"] = np.mean(peak_values, axis=0)
        print(f"Found peaks at positions: {peak_positions} with values: {peak_values}")
        return data
    
# class FindAmplitudeGradDesc:
#     """
#     A class to find the amplitude of a signal, local maxima using gradient descent. 
#     find peaks seems more appropriate for this task, but this is a placeholder for future work.
#     ""

#     def __init__(self, peak_position=257):
#         self.peak_position = peak_position 

#     def process(self, data, numiterations=1000, alpha=0.01):
#         fx = data[:, self.peak_position]
#         x = np.linspace(0, len(fx) - 1, len(fx))

#         # Interpolate function
#         f_interp = interp1d(x, fx, kind='cubic', fill_value='extrapolate')

#         # Compute and interpolate gradient
#         grad_fx = np.gradient(fx, x)
#         grad_interp = interp1d(x, grad_fx, kind='cubic', fill_value='extrapolate')

#         x_current = x[0]
#         for _ in range(numiterations):
#             grad = grad_interp(x_current)
#             x_next = x_current - alpha * grad

#             if f_interp(x_next) < f_interp(x_current):
#                 x_current = x_next
#             else:
#                 alpha *= 0.5

#         return x_current, f_interp(x_current)