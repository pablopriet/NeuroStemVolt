from .base import Processor
import numpy as np
from scipy.signal import find_peaks
from PyQt5.QtCore import QSettings

class FindAmplitudeMultiple(Processor):
    def __init__(self, peak_position, min_prominence=0.05, min_decay_ratio=0.1, 
                 rise_window_sec=1.0, decay_window_sec=20.0, max_peaks=10):
        """
        Initialize FindAmplitudeMultiple for spontaneous signal analysis.
        
        Args:
            peak_position: Column index for peak detection
            min_prominence: Minimum peak prominence (lowered for spontaneous)
            min_decay_ratio: Minimum decay ratio from peak
            rise_window_sec: Time window in seconds to check for rise (reduced for spontaneous)
            decay_window_sec: Time window in seconds to check for decay (reduced for spontaneous)
            max_peaks: Maximum number of peaks to return
        """
        self.peak_position = peak_position
        self.min_prominence = min_prominence
        self.min_decay_ratio = min_decay_ratio
        self.rise_window_sec = rise_window_sec
        self.decay_window_sec = decay_window_sec
        self.max_peaks = max_peaks

    def _find_adaptive_time_windows(self, fx, peak_idx, freq):
        """
        Dynamically find optimal rise and decay windows for spontaneous signals.
        Uses shorter time windows compared to stimulated signals.
        """
        # Adjusted minimum and maximum window constraints for spontaneous signals
        MIN_RISE_TIME_SEC = 0.2   # Reduced from 0.5s - spontaneous events can be faster
        MAX_RISE_TIME_SEC = 3.0   # Reduced from 5.0s
        MIN_DECAY_TIME_SEC = 1.0  # Reduced from 2.0s
        MAX_DECAY_TIME_SEC = 10.0 # Reduced from 20.0s

        # Convert to samples
        min_rise_samples = max(3, int(MIN_RISE_TIME_SEC * freq))
        max_rise_samples = int(MAX_RISE_TIME_SEC * freq)
        min_decay_samples = max(5, int(MIN_DECAY_TIME_SEC * freq))
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
        """Find the optimal rise window for spontaneous signals."""
        peak_val = fx[peak_idx]
        start_search = max(0, peak_idx - max_samples)

        # Find baseline level
        early_points = fx[start_search:max(1, start_search + min_samples // 2)]
        if len(early_points) > 0:
            baseline = np.median(early_points)
        else:
            baseline = 0

        # More sensitive rise threshold for spontaneous signals
        rise_threshold = baseline + 0.05 * (peak_val - baseline)  # Reduced from 0.1

        # Find where signal first crosses the rise threshold
        rise_start_idx = peak_idx
        for i in range(peak_idx - 1, start_search - 1, -1):
            if fx[i] < rise_threshold:
                rise_start_idx = i
                break

        rise_window = max(min_samples, peak_idx - rise_start_idx)
        rise_window = min(max_samples, rise_window)

        return rise_window

    def _find_decay_window(self, fx, peak_idx, peak_val, min_samples, max_samples):
        """Find the optimal decay window for spontaneous signals."""
        end_search = min(len(fx), peak_idx + max_samples)

        if end_search <= peak_idx + min_samples:
            return min_samples

        # Find baseline
        far_start = min(len(fx) - 5, peak_idx + max_samples - 5)
        if far_start < len(fx):
            far_points = fx[far_start:len(fx)]
            if len(far_points) > 2:
                baseline = np.median(far_points)
            else:
                baseline = fx[-1] if len(fx) > 0 else 0
        else:
            baseline = fx[min(len(fx) - 1, peak_idx + min_samples)]

        # Adjusted decay thresholds for spontaneous signals
        decay_70_threshold = baseline + 0.7 * (peak_val - baseline)  # 70% instead of 90%
        decay_30_threshold = baseline + 0.3 * (peak_val - baseline)  # 30% instead of 50%

        # Find threshold crossings
        decay_70_idx = None
        decay_30_idx = None

        for i in range(peak_idx + 1, end_search):
            if i >= len(fx):
                break

            if decay_70_idx is None and fx[i] <= decay_70_threshold:
                decay_70_idx = i
            if decay_30_idx is None and fx[i] <= decay_30_threshold:
                decay_30_idx = i
                break

        # Determine decay window
        if decay_30_idx is not None:
            decay_window = min(max_samples, (decay_30_idx - peak_idx) + 5)
        elif decay_70_idx is not None:
            decay_window = min(max_samples, (decay_70_idx - peak_idx) * 2)
        else:
            decay_window = self._find_decay_by_slope(fx, peak_idx, min_samples, max_samples)

        decay_window = max(min_samples, decay_window)
        return decay_window

    def _find_decay_by_slope(self, fx, peak_idx, min_samples, max_samples):
        """Fallback: find decay window by slope analysis for spontaneous signals."""
        end_search = min(len(fx), peak_idx + max_samples)

        if end_search <= peak_idx + min_samples:
            return min_samples

        window_size = 3  # Smaller window for spontaneous signals
        slopes = []

        for i in range(peak_idx + window_size, end_search - window_size):
            if i + window_size >= len(fx):
                break
            slope = (fx[i + window_size] - fx[i - window_size]) / (2 * window_size)
            slopes.append((i, slope))

        if not slopes:
            return min_samples

        # More sensitive slope threshold for spontaneous signals
        slope_threshold = 0.002

        for i, slope in slopes:
            if abs(slope) < slope_threshold:
                decay_window = min(max_samples, i - peak_idx + 5)
                return max(min_samples, decay_window)

        return min(max_samples, min_samples * 2)

    def _validate_peak_with_adaptive_windows(self, fx, peak_idx, freq):
        """Validate peak using adaptive time windows for spontaneous signals."""
        rise_window, decay_window, rise_time_sec, decay_time_sec = self._find_adaptive_time_windows(fx, peak_idx, freq)

        decay_valid = self._validate_peak_decay_adaptive(fx, peak_idx, decay_window)
        rise_valid = self._validate_left_side_rise_adaptive(fx, peak_idx, rise_window)

        # More lenient window validation for spontaneous signals
        window_valid = (rise_time_sec >= 0.2 and decay_time_sec >= 1.0)

        return decay_valid and rise_valid and window_valid, {
            'rise_window_samples': rise_window,
            'decay_window_samples': decay_window,
            'rise_time_sec': rise_time_sec,
            'decay_time_sec': decay_time_sec
        }

    def _validate_peak_decay_adaptive(self, fx, peak_idx, decay_window):
        """Validate decay for spontaneous signals."""
        peak_val = fx[peak_idx]
        end_idx = min(len(fx), peak_idx + decay_window)

        if end_idx <= peak_idx + 3:  # Reduced minimum points
            return False

        decay_region = fx[peak_idx:end_idx]
        x_vals = np.arange(len(decay_region))
        
        if len(x_vals) < 3:
            return False

        try:
            coeffs = np.polyfit(x_vals, decay_region, 1)
            slope = coeffs[0]

            if slope >= 0:
                return False

            # More lenient decay requirement for spontaneous signals
            end_val = np.median(decay_region[-3:])  # Last 3 points
            decay_ratio = (peak_val - end_val) / peak_val if peak_val != 0 else 0

            return decay_ratio >= 0.1  # Reduced from 0.2

        except:
            return False

    def _validate_left_side_rise_adaptive(self, fx, peak_idx, rise_window):
        """Validate rise for spontaneous signals."""
        peak_val = fx[peak_idx]
        start_idx = max(0, peak_idx - rise_window)

        if peak_idx - start_idx < 3:  # Reduced minimum points
            return False

        rise_region = fx[start_idx:peak_idx + 1]
        x_vals = np.arange(len(rise_region))

        try:
            coeffs = np.polyfit(x_vals, rise_region, 1)
            slope = coeffs[0]

            if slope <= 0:
                return False

            # More lenient rise requirement
            start_val = np.median(rise_region[:3])  # First 3 points
            rise_amplitude = peak_val - start_val

            return rise_amplitude >= 0.03  # Reduced from 0.05

        except:
            return False

    def process(self, data, context=None):
        """
        Enhanced process method for multiple peak detection in spontaneous signals.
        """
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)

        fx = data[:, self.peak_position]

        # More sensitive initial peak detection for spontaneous signals
        peaks, properties = find_peaks(fx,
                                       prominence=self.min_prominence,  # Lower threshold
                                       distance=max(5, freq // 2),      # 0.5 second minimum distance
                                       height=0.02,                     # Lower height threshold
                                       width=2)                         # Narrower minimum width

        print(f"Initial peaks found for spontaneous analysis: {len(peaks)}")

        # Validate all peaks
        valid_peaks = []
        valid_values = []
        peak_metadata = []

        for peak_idx in peaks:
            peak_val = fx[peak_idx]

            is_valid, window_info = self._validate_peak_with_adaptive_windows(fx, peak_idx, freq)

            if is_valid:
                valid_peaks.append(peak_idx)
                valid_values.append(peak_val)
                peak_metadata.append(window_info)
                print(f"Valid spontaneous peak: idx={peak_idx}, val={peak_val:.3f}, "
                      f"rise={window_info['rise_time_sec']:.1f}s, "
                      f"decay={window_info['decay_time_sec']:.1f}s")

        # Sort by amplitude and take top peaks
        if len(valid_peaks) > 0:
            # Sort by amplitude (descending)
            sorted_indices = np.argsort(valid_values)[::-1]
            
            # Limit to max_peaks
            num_peaks_to_keep = min(self.max_peaks, len(valid_peaks))
            
            final_peaks = [valid_peaks[i] for i in sorted_indices[:num_peaks_to_keep]]
            final_values = [valid_values[i] for i in sorted_indices[:num_peaks_to_keep]]
            final_metadata = [peak_metadata[i] for i in sorted_indices[:num_peaks_to_keep]]
            
            # Sort by time (position) for final output
            time_sorted_indices = np.argsort(final_peaks)
            final_peaks = [final_peaks[i] for i in time_sorted_indices]
            final_values = [final_values[i] for i in time_sorted_indices]
            final_metadata = [final_metadata[i] for i in time_sorted_indices]

            print(f"Found {len(final_peaks)} valid spontaneous peaks")
            for i, (pos, val) in enumerate(zip(final_peaks, final_values)):
                print(f"  Peak {i+1}: position={pos} (time={pos/freq:.1f}s), amplitude={val:.3f}")

        else:
            final_peaks = []
            final_values = []
            final_metadata = []
            print("No valid spontaneous peaks found.")

        # Update context for multiple peaks
        if context is not None:
            context['peak_amplitude_positions'] = final_peaks
            context['peak_amplitude_values'] = final_values
            context['num_peaks_detected'] = len(final_peaks)
            
            # Store metadata for all peaks
            context['all_peak_metadata'] = final_metadata
            
            # For backwards compatibility, store the highest amplitude peak as primary
            if len(final_peaks) > 0:
                # Find the peak with highest amplitude
                max_amp_idx = np.argmax(final_values)
                context['primary_peak_position'] = final_peaks[max_amp_idx]
                context['primary_peak_value'] = final_values[max_amp_idx]
                
                # Store primary peak's decay regions
                primary_metadata = final_metadata[max_amp_idx]
                peak_position = final_peaks[max_amp_idx]
                
                rise_samples = primary_metadata['rise_window_samples']
                decay_samples = primary_metadata['decay_window_samples']

                left_start = max(0, peak_position - rise_samples)
                context['decay_left_region'] = {
                    'indices': list(range(left_start, peak_position)),
                    'values': fx[left_start:peak_position].tolist(),
                    'time_window_sec': primary_metadata['rise_time_sec'],
                    'adaptive': True
                }

                right_end = min(len(fx), peak_position + decay_samples + 1)
                context['decay_right_region'] = {
                    'indices': list(range(peak_position + 1, right_end)),
                    'values': fx[peak_position + 1:right_end].tolist(),
                    'time_window_sec': primary_metadata['decay_time_sec'],
                    'adaptive': True
                }

                context['decay_validation_params'] = {
                    'adaptive_windows': True,
                    'rise_window_sec': primary_metadata['rise_time_sec'],
                    'decay_window_sec': primary_metadata['decay_time_sec'],
                    'rise_window_samples': rise_samples,
                    'decay_window_samples': decay_samples,
                    'acquisition_frequency': freq,
                    'peak_passed_validation': True,
                    'validation_type': 'adaptive_spontaneous_multiple_peaks',
                    'signal_type': 'spontaneous',
                    'multiple_peaks': True
                }

        return data
