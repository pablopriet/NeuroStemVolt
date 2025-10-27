from .base import Processor
import numpy as np
from scipy.signal import find_peaks
from PyQt5.QtCore import QSettings
from scipy.optimize import curve_fit

class FindAmplitudeLegacy(Processor):
    """
    Detect the most prominent peak in the I-T profile at a given peak position.

    Stores the position and value of the peak in the context dictionary.

    Args:
        peak_position (int): Index for voltage sweep peak (default 257).
    """
    def __init__(self, peak_position=257):
        self.peak_position = peak_position

    def process(self, data, context=None):
        """
        Detect the highest peak in the I-T trace and update context with its value and location.

        Args:
            data (np.ndarray): 2D FSCV array.
            context (dict): Optional metadata dictionary.

        Returns:
            np.ndarray: Unchanged data array.
        """
        fx = data[:, self.peak_position]
        peaks, _ = find_peaks(fx, 
                              #prominence=0.2, 
                              #distance=10, 
                              height=0.1)
        # Get the actual peak values
        peak_values = fx[peaks]

        if len(peak_values) > 0:
            max_peak_idx = np.argmax(peak_values)
            peak_position = peaks[max_peak_idx]
            peak_value = peak_values[max_peak_idx]
            print("Peak position:", peak_position)
            print("Peak value:", peak_value)
        else:
            print("No peaks found.")

        # Update context with peak information
        if context is not None:
            context['peak_amplitude_positions'] = peak_position
            context['peak_amplitude_values'] = peak_value
        print(f"Found peaks at positions: {peak_position} with values: {peak_values}")
        return data
    
class FindAmplitude(Processor):
    def __init__(self, peak_position, min_prominence=0.1, min_decay_ratio=0.1, 
             rise_window_sec=2.0, decay_window_sec=40.0):
        """
        Initialize FindAmplitude with realistic time windows for neurotransmitter signals.
        
        Args:
            peak_position: Column index for peak detection
            min_prominence: Minimum peak prominence
            min_decay_ratio: Minimum decay ratio from peak
            rise_window_sec: Time window in seconds to check for rise (default 2.0s)
            decay_window_sec: Time window in seconds to check for decay (default 40.0s)
        """
        self.peak_position = peak_position
        self.min_prominence = min_prominence
        self.min_decay_ratio = min_decay_ratio
        self.rise_window_sec = rise_window_sec
        self.decay_window_sec = decay_window_sec


    def _find_adaptive_time_windows(self, fx, peak_idx, freq):
        """
        Dynamically find optimal rise and decay windows for each peak.
        Ensures windows are long enough to represent real neurotransmitter events.
        """
        # Minimum and maximum window constraints (in seconds)
        MIN_RISE_TIME_SEC = 0.5  # At least 0.5 seconds for biological rise
        MAX_RISE_TIME_SEC = 5.0  # Maximum 5 seconds for rise
        MIN_DECAY_TIME_SEC = 2.0  # At least 2 seconds for meaningful decay
        MAX_DECAY_TIME_SEC = 20.0  # Maximum 20 seconds for decay

        # Convert to samples
        min_rise_samples = max(5, int(MIN_RISE_TIME_SEC * freq))
        max_rise_samples = int(MAX_RISE_TIME_SEC * freq)
        min_decay_samples = max(10, int(MIN_DECAY_TIME_SEC * freq))
        max_decay_samples = int(MAX_DECAY_TIME_SEC * freq)

        peak_val = fx[peak_idx]

        # Find adaptive rise window
        rise_window = self._find_rise_window(fx, peak_idx, min_rise_samples, max_rise_samples)

        # Find adaptive decay window
        decay_window = self._find_decay_window(fx, peak_idx, peak_val, min_decay_samples, max_decay_samples)

        # Convert back to seconds for reporting
        rise_time_sec = rise_window / freq
        decay_time_sec = decay_window / freq

        print(f"Adaptive windows for peak at {peak_idx}: rise={rise_time_sec:.1f}s, decay={decay_time_sec:.1f}s")

        return rise_window, decay_window, rise_time_sec, decay_time_sec

    def _find_rise_window(self, fx, peak_idx, min_samples, max_samples):
        """
        Find the optimal rise window by looking for the start of the rise phase.
        """
        peak_val = fx[peak_idx]

        # Look backwards from peak to find where rise begins
        start_search = max(0, peak_idx - max_samples)

        # Find baseline level (use median of early points)
        early_points = fx[start_search:max(1, start_search + min_samples // 2)]
        if len(early_points) > 0:
            baseline = np.median(early_points)
        else:
            baseline = 0

        # Define rise threshold (10% of peak amplitude above baseline)
        rise_threshold = baseline + 0.1 * (peak_val - baseline)

        # Find where signal first crosses the rise threshold
        rise_start_idx = peak_idx
        for i in range(peak_idx - 1, start_search - 1, -1):
            if fx[i] < rise_threshold:
                rise_start_idx = i
                break

        # Calculate rise window, ensuring it meets minimum requirements
        rise_window = max(min_samples, peak_idx - rise_start_idx)
        rise_window = min(max_samples, rise_window)

        return rise_window

    def _find_decay_window(self, fx, peak_idx, peak_val, min_samples, max_samples):
        """
        Find the optimal decay window by analyzing the decay profile.
        """
        # Look forward from peak
        end_search = min(len(fx), peak_idx + max_samples)

        if end_search <= peak_idx + min_samples:
            return min_samples

        # Find baseline level (use median of far points if available)
        far_start = min(len(fx) - 10, peak_idx + max_samples - 10)
        if far_start < len(fx):
            far_points = fx[far_start:len(fx)]
            if len(far_points) > 3:
                baseline = np.median(far_points)
            else:
                baseline = fx[-1] if len(fx) > 0 else 0
        else:
            baseline = fx[min(len(fx) - 1, peak_idx + min_samples)]

        # Define decay thresholds
        decay_90_threshold = baseline + 0.9 * (peak_val - baseline)  # 90% of peak
        decay_50_threshold = baseline + 0.5 * (peak_val - baseline)  # 50% of peak
        decay_20_threshold = baseline + 0.2 * (peak_val - baseline)  # 20% of peak

        # Find where signal crosses these thresholds
        decay_50_idx = None
        decay_20_idx = None

        for i in range(peak_idx + 1, end_search):
            if i >= len(fx):
                break

            if decay_50_idx is None and fx[i] <= decay_50_threshold:
                decay_50_idx = i
            if decay_20_idx is None and fx[i] <= decay_20_threshold:
                decay_20_idx = i
                break

        # Determine decay window based on threshold crossings
        if decay_20_idx is not None:
            # Use 20% threshold crossing plus some buffer
            decay_window = min(max_samples, (decay_20_idx - peak_idx) + 10)
        elif decay_50_idx is not None:
            # Use 50% threshold crossing with larger buffer
            decay_window = min(max_samples, (decay_50_idx - peak_idx) * 2)
        else:
            # Fallback to analyzing slope changes
            decay_window = self._find_decay_by_slope(fx, peak_idx, min_samples, max_samples)

        # Ensure minimum requirements
        decay_window = max(min_samples, decay_window)

        return decay_window

    def _find_decay_by_slope(self, fx, peak_idx, min_samples, max_samples):
        """
        Fallback method: find decay window by analyzing slope changes.
        """
        end_search = min(len(fx), peak_idx + max_samples)

        if end_search <= peak_idx + min_samples:
            return min_samples

        # Calculate moving average of slopes
        window_size = 5
        slopes = []

        for i in range(peak_idx + window_size, end_search - window_size):
            if i + window_size >= len(fx):
                break
            # Calculate slope over small window
            slope = (fx[i + window_size] - fx[i - window_size]) / (2 * window_size)
            slopes.append((i, slope))

        if not slopes:
            return min_samples

        # Find where slope becomes very small (decay nearly complete)
        slope_threshold = 0.001  # Very small slope indicates flat region

        for i, slope in slopes:
            if abs(slope) < slope_threshold:
                decay_window = min(max_samples, i - peak_idx + 10)  # Add small buffer
                return max(min_samples, decay_window)

        # If no flat region found, use a reasonable default
        return min(max_samples, min_samples * 2)

    def _validate_peak_with_adaptive_windows(self, fx, peak_idx, freq):
        """
        Validate peak using adaptive time windows.
        """
        # Get adaptive windows
        rise_window, decay_window, rise_time_sec, decay_time_sec = self._find_adaptive_time_windows(fx, peak_idx, freq)

        # Validate using the adaptive windows
        decay_valid = self._validate_peak_decay_adaptive(fx, peak_idx, decay_window)
        rise_valid = self._validate_left_side_rise_adaptive(fx, peak_idx, rise_window)

        # Additional validation: ensure windows are reasonable
        window_valid = (rise_time_sec >= 0.5 and decay_time_sec >= 2.0)

        return decay_valid and rise_valid and window_valid, {
            'rise_window_samples': rise_window,
            'decay_window_samples': decay_window,
            'rise_time_sec': rise_time_sec,
            'decay_time_sec': decay_time_sec
        }

    def _validate_peak_decay_adaptive(self, fx, peak_idx, decay_window):
        """
        Validate decay using adaptive window.
        """
        peak_val = fx[peak_idx]

        # Check decay over the adaptive window
        end_idx = min(len(fx), peak_idx + decay_window)

        if end_idx <= peak_idx + 5:  # Need at least 5 points
            return False

        # Check that signal generally decreases
        decay_region = fx[peak_idx:end_idx]

        # Use linear regression to check overall trend
        x_vals = np.arange(len(decay_region))
        if len(x_vals) < 3:
            return False

        try:
            # Fit linear trend
            coeffs = np.polyfit(x_vals, decay_region, 1)
            slope = coeffs[0]

            # Should have negative slope for decay
            if slope >= 0:
                return False

            # Check amplitude decrease
            end_val = np.median(decay_region[-5:])  # Average of last 5 points
            decay_ratio = (peak_val - end_val) / peak_val if peak_val != 0 else 0

            # Should decay by at least 20% over the window
            return decay_ratio >= 0.2

        except:
            return False

    def _validate_left_side_rise_adaptive(self, fx, peak_idx, rise_window):
        """
        Validate rise using adaptive window.
        """
        peak_val = fx[peak_idx]

        # Check rise over the adaptive window
        start_idx = max(0, peak_idx - rise_window)

        if peak_idx - start_idx < 5:  # Need at least 5 points
            return False

        rise_region = fx[start_idx:peak_idx + 1]

        # Use linear regression to check overall trend
        x_vals = np.arange(len(rise_region))

        try:
            # Fit linear trend
            coeffs = np.polyfit(x_vals, rise_region, 1)
            slope = coeffs[0]

            # Should have positive slope for rise
            if slope <= 0:
                return False

            # Check amplitude increase
            start_val = np.median(rise_region[:5])  # Average of first 5 points
            rise_amplitude = peak_val - start_val

            # Should rise by reasonable amount
            return rise_amplitude >= 0.05  # Minimum 0.05 units rise

        except:
            return False

    def process(self, data, context=None):
        """
        Enhanced process method with adaptive time windows for each peak
        """

        # Get acquisition frequency from settings
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)

        fx = data[:, self.peak_position]

        # Initial peak detection with relaxed parameters for adaptive validation
        peaks, properties = find_peaks(fx,
                                       prominence=self.min_prominence,
                                       distance=max(10, freq),  # 1 second minimum distance
                                       height=0.03,  # Slightly higher threshold
                                       width=3)  # Minimum width to avoid spikes

        if len(peaks) == 0:
            print("No peaks found in the initial detection.")
            print("Using fallback logic to find peaks...")
            # Fallback logic: use the maximum value as the peak
            peak_position = np.argmax(fx)
            peak_value = fx[peak_position]
            print(f"Fallback peak at position: {peak_position} with value: {peak_value:.3f}")
            context['peak_amplitude_positions'] = peak_position
            context['peak_amplitude_values'] = peak_value
            return data

        print(f"Initial peaks found: {len(peaks)}")
        print(f"Using adaptive time windows for validation")

        # Validate peaks using adaptive windows
        valid_peaks = []
        valid_values = []
        peak_metadata = []

        for peak_idx in peaks:
            peak_val = fx[peak_idx]

            # Use adaptive validation
            is_valid, window_info = self._validate_peak_with_adaptive_windows(fx, peak_idx, freq)

            if is_valid:
                valid_peaks.append(peak_idx)
                valid_values.append(peak_val)
                peak_metadata.append(window_info)
                print(f"Valid peak: idx={peak_idx}, val={peak_val:.3f}, "
                      f"rise={window_info['rise_time_sec']:.1f}s, "
                      f"decay={window_info['decay_time_sec']:.1f}s")
            else:
                print(f"Invalid peak: idx={peak_idx}, val={peak_val:.3f} "
                      f"(failed adaptive validation)")

        # Convert to numpy arrays
        valid_peaks = np.array(valid_peaks)
        valid_values = np.array(valid_values)

        # Find the dominant peak
        peak_position = 0
        peak_value = 0
        selected_metadata = None

        if len(valid_values) > 0:
            max_peak_idx = np.argmax(valid_values)
            peak_position = valid_peaks[max_peak_idx]
            peak_value = valid_values[max_peak_idx]
            selected_metadata = peak_metadata[max_peak_idx]
            print(f"Selected peak at position: {peak_position} (time: {peak_position / freq:.1f}s)")
            print(f"Adaptive windows: rise={selected_metadata['rise_time_sec']:.1f}s, "
                  f"decay={selected_metadata['decay_time_sec']:.1f}s")
        else:
            print("No valid peaks found with adaptive validation.")
            # NEW fallback here:
            peak_position = np.argmax(fx)
            peak_value = fx[peak_position]
            print(f"Fallback peak at position: {peak_position} with value: {peak_value:.3f}")
            if context is not None:
                context['peak_amplitude_positions'] = peak_position
                context['peak_amplitude_values'] = peak_value

        # Enhanced context with adaptive window information
        if context is not None and selected_metadata is not None:
            context['peak_amplitude_positions'] = peak_position
            context['peak_amplitude_values'] = peak_value

            # Add adaptive window information
            rise_samples = selected_metadata['rise_window_samples']
            decay_samples = selected_metadata['decay_window_samples']

            # Left side (rise) region
            left_start = max(0, peak_position - rise_samples)
            context['decay_left_region'] = {
                'indices': list(range(left_start, peak_position)),
                'values': fx[left_start:peak_position].tolist(),
                'time_window_sec': selected_metadata['rise_time_sec'],
                'adaptive': True
            }

            # Right side (decay) region
            right_end = min(len(fx), peak_position + decay_samples + 1)
            context['decay_right_region'] = {
                'indices': list(range(peak_position + 1, right_end)),
                'values': fx[peak_position + 1:right_end].tolist(),
                'time_window_sec': selected_metadata['decay_time_sec'],
                'adaptive': True
            }

            # Enhanced validation parameters
            context['decay_validation_params'] = {
                'adaptive_windows': True,
                'rise_window_sec': selected_metadata['rise_time_sec'],
                'decay_window_sec': selected_metadata['decay_time_sec'],
                'rise_window_samples': rise_samples,
                'decay_window_samples': decay_samples,
                'acquisition_frequency': freq,
                'peak_passed_validation': is_valid,
                'validation_type': 'adaptive_neurotransmitter_profile'
            }

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