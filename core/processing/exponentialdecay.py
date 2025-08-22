from .base import Processor
import numpy as np
from scipy.optimize import curve_fit

class ExponentialFitting(Processor):
    """
    Fit an exponential decay curve to the I-T profile after the peak.

    Extracts parameters A, tau, and C from the model:
        I(t) = A * exp(-t / tau) + C
    Also computes the half-life `t_half`.

    Methods:
        process(data, peak_position, context): Applies fitting and stores parameters.
    """
    def process(self, data, peak_position=257, context=None):
        """
        Fit exponential decay to the I-T trace starting at the detected peak.

        Args:
            data (np.ndarray): 2D FSCV array (voltage × time).
            peak_position (int): Index for peak current extraction.
            context (dict): Must contain 'peak_amplitude_positions'.

        Returns:
            np.ndarray: Original input data (unchanged).
        """
        if context is not None:
            if "peak_amplitude_positions" in context:
                peak_amplitude_position = context["peak_amplitude_positions"]

                # Robust empty check
                if (isinstance(peak_amplitude_position, (list, np.ndarray)) and np.size(peak_amplitude_position) == 0):
                    print("No peak found, skipping exponential fitting.")
                    context["exponential fitting parameters"] = {
                        "A": 0,
                        "tau": 0,
                        "C": 0,
                        "t_half": 0,
                    }
                    return data

                # Robust conversion to int
                if isinstance(peak_amplitude_position, (list, np.ndarray)):
                    if np.size(peak_amplitude_position) > 1:
                        peak_amplitude_position = int(np.mean(peak_amplitude_position))
                    else:
                        peak_amplitude_position = int(np.ravel(peak_amplitude_position)[0])
                else:
                    peak_amplitude_position = int(peak_amplitude_position)

                IT_profile = data[:, peak_position]
                y = IT_profile[peak_amplitude_position:]
                t = np.arange(peak_amplitude_position, peak_amplitude_position + len(y))
                print(y.shape, t.shape)

                # Fitting the exponential decay
                A0 = y[0] - y[-1]          # Approx amplitude
                tau0 = (t[-1] - t[0]) / 2  # Mid-range guess for decay constant
                C0 = y[-1]                 # Final value as baseline

                # Fit
                popt, pcov = curve_fit(exp_decay, t, y, p0=[A0, tau0, C0])

                A, tau, C = popt
                t_half = np.log(2) * tau
                context["exponential fitting parameters"] = {
                    "A": A,
                    "tau": tau,
                    "C": C,
                    "t_half": t_half,
                }
                print(f"Fitted parameters: A={A}, tau={tau}, C={C}")
        return data
    
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def exp_decay_k(t, A, k, C):
    return A * np.exp(-k * t) + C