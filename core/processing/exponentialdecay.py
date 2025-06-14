from .base import Processor
import numpy as np
from scipy.optimize import curve_fit

class ExponentialFitting(Processor):
    def process(self, data, peak_position=257, context = None):
        if context is not None:
            if "peak_amplitude_positions" in context: 
                peak_amplitude_position = context["peak_amplitude_positions"]
                IT_profile = data[:, peak_position]
                if len(peak_amplitude_position) > 0:
                    peak_amplitude_position = int(np.mean(peak_amplitude_position, axis=0))
                else:
                    peak_amplitude_position = int(peak_amplitude_position)
                y = IT_profile[peak_amplitude_position:]
                t = np.arange(peak_amplitude_position,peak_amplitude_position + len(y))
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