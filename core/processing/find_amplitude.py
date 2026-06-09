from .base import Processor
import numpy as np
from scipy.signal import find_peaks
from PyQt5.QtCore import QSettings


class FindAmplitudeLegacy(Processor):
    """Legacy single-peak detector — kept for reference. Use FindAmplitude instead."""

    def __init__(self, peak_position=257):
        self.peak_position = peak_position

    def process(self, data, context=None):
        fx = data[:, self.peak_position]
        peaks, _ = find_peaks(fx, height=0.1)
        peak_values = fx[peaks]

        if len(peak_values) > 0:
            max_peak_idx = np.argmax(peak_values)
            peak_position = int(peaks[max_peak_idx])
            peak_value = float(peak_values[max_peak_idx])
        else:
            peak_position = int(np.argmax(fx))
            peak_value = float(fx[peak_position])

        if context is not None:
            context['peak_amplitude_positions'] = peak_position
            context['peak_amplitude_values'] = peak_value
        return data


class FindAmplitude(Processor):
    """
    Detect the most prominent valid peak in the I-T profile.

    If no valid peak is found the signal maximum is used as a fallback and a
    warning is added to ``context['processing_warnings']``, which the
    processing-warnings dialog displays to the user.

    Args:
        peak_position (int): Column index of the voltage step to analyse.
        prominence_fraction (float): Minimum prominence as a fraction of the
            signal's 5–95th-percentile range (default 0.10 = 10 %).
        min_height_na (float): Absolute nA floor — peaks below this are
            ignored regardless of prominence (default 0.03 nA).
        rise_window_sec (float): Hint for rise-window search (default 2.0 s).
        decay_window_sec (float): Hint for decay-window search (default 40.0 s).
    """

    # ── Detection constants ──────────────────────────────────────────────────
    DEFAULT_PROMINENCE_FRACTION = 0.10
    DEFAULT_MIN_HEIGHT_NA       = 0.03
    MIN_PEAK_DISTANCE_SEC       = 1.0   # minimum gap between candidate peaks (s)
    MIN_PEAK_WIDTH_SCANS        = 3     # minimum scan width — rejects single-scan spikes

    # ── Adaptive validation window limits ────────────────────────────────────
    MIN_RISE_TIME_SEC  = 0.5
    MAX_RISE_TIME_SEC  = 5.0
    MIN_DECAY_TIME_SEC = 2.0
    MAX_DECAY_TIME_SEC = 20.0

    # ── Signal-range percentiles for normalisation ───────────────────────────
    SIGNAL_PCT_LOW  = 5
    SIGNAL_PCT_HIGH = 95

    def __init__(self, peak_position, prominence_fraction=0.10, min_height_na=0.03,
                 rise_window_sec=2.0, decay_window_sec=40.0):
        self.peak_position       = peak_position
        self.prominence_fraction = prominence_fraction
        self.min_height_na       = min_height_na
        self.rise_window_sec     = rise_window_sec
        self.decay_window_sec    = decay_window_sec

    # ── Private helpers ──────────────────────────────────────────────────────

    def _find_adaptive_time_windows(self, fx, peak_idx, freq):
        min_rise  = max(5,  int(self.MIN_RISE_TIME_SEC  * freq))
        max_rise  =         int(self.MAX_RISE_TIME_SEC  * freq)
        min_decay = max(10, int(self.MIN_DECAY_TIME_SEC * freq))
        max_decay =         int(self.MAX_DECAY_TIME_SEC * freq)

        rise_w  = self._find_rise_window(fx, peak_idx, min_rise, max_rise)
        decay_w = self._find_decay_window(fx, peak_idx, fx[peak_idx], min_decay, max_decay)

        return rise_w, decay_w, rise_w / freq, decay_w / freq

    def _find_rise_window(self, fx, peak_idx, min_samples, max_samples):
        peak_val     = fx[peak_idx]
        start_search = max(0, peak_idx - max_samples)
        early        = fx[start_search: max(1, start_search + min_samples // 2)]
        baseline     = float(np.median(early)) if len(early) > 0 else 0.0
        threshold    = baseline + 0.1 * (peak_val - baseline)

        rise_start = peak_idx
        for i in range(peak_idx - 1, start_search - 1, -1):
            if fx[i] < threshold:
                rise_start = i
                break

        return max(min_samples, min(max_samples, peak_idx - rise_start))

    def _find_decay_window(self, fx, peak_idx, peak_val, min_samples, max_samples):
        end_search = min(len(fx), peak_idx + max_samples)
        if end_search <= peak_idx + min_samples:
            return min_samples

        far_start  = min(len(fx) - 10, peak_idx + max_samples - 10)
        far_points = fx[far_start: len(fx)]
        baseline   = float(np.median(far_points)) if len(far_points) > 3 else (
                     float(fx[-1]) if len(fx) > 0 else 0.0)

        t50 = baseline + 0.5 * (peak_val - baseline)
        t20 = baseline + 0.2 * (peak_val - baseline)
        idx50 = idx20 = None

        for i in range(peak_idx + 1, end_search):
            if i >= len(fx):
                break
            if idx50 is None and fx[i] <= t50:
                idx50 = i
            if idx20 is None and fx[i] <= t20:
                idx20 = i
                break

        if idx20 is not None:
            w = min(max_samples, (idx20 - peak_idx) + 10)
        elif idx50 is not None:
            w = min(max_samples, (idx50 - peak_idx) * 2)
        else:
            w = self._find_decay_by_slope(fx, peak_idx, min_samples, max_samples)

        return max(min_samples, w)

    def _find_decay_by_slope(self, fx, peak_idx, min_samples, max_samples):
        end_search  = min(len(fx), peak_idx + max_samples)
        window_size = 5
        for i in range(peak_idx + window_size, end_search - window_size):
            if i + window_size >= len(fx):
                break
            slope = (fx[i + window_size] - fx[i - window_size]) / (2 * window_size)
            if abs(slope) < 0.001:
                return max(min_samples, min(max_samples, i - peak_idx + 10))
        return min(max_samples, min_samples * 2)

    def _validate_peak(self, fx, peak_idx, freq):
        """Validate a candidate peak using adaptive rise/decay windows."""
        rise_w, decay_w, rise_sec, decay_sec = self._find_adaptive_time_windows(
            fx, peak_idx, freq)

        decay_ok  = self._check_decay(fx, peak_idx, decay_w)
        rise_ok   = self._check_rise(fx, peak_idx, rise_w)
        window_ok = rise_sec >= self.MIN_RISE_TIME_SEC and decay_sec >= self.MIN_DECAY_TIME_SEC

        return (decay_ok and rise_ok and window_ok), {
            'rise_window_samples':  rise_w,
            'decay_window_samples': decay_w,
            'rise_time_sec':        rise_sec,
            'decay_time_sec':       decay_sec,
        }

    def _check_decay(self, fx, peak_idx, decay_window):
        peak_val = fx[peak_idx]
        end_idx  = min(len(fx), peak_idx + decay_window)
        if end_idx <= peak_idx + 5:
            return False
        region = fx[peak_idx: end_idx]
        if len(region) < 3:
            return False
        try:
            slope = np.polyfit(np.arange(len(region)), region, 1)[0]
            if slope >= 0:
                return False
            end_val = float(np.median(region[-5:]))
            return (peak_val - end_val) / peak_val >= 0.2 if peak_val != 0 else False
        except Exception:
            return False

    def _check_rise(self, fx, peak_idx, rise_window):
        peak_val  = fx[peak_idx]
        start_idx = max(0, peak_idx - rise_window)
        if peak_idx - start_idx < 5:
            return False
        region = fx[start_idx: peak_idx + 1]
        try:
            slope = np.polyfit(np.arange(len(region)), region, 1)[0]
            if slope <= 0:
                return False
            return float(peak_val - np.median(region[:5])) >= 0.05
        except Exception:
            return False

    def process(self, data, context=None):
        """
        Detect the dominant peak in the I-T trace and write metadata to context.
        """
        freq = QSettings("HashemiLab", "NeuroStemVolt").value(
            "acquisition_frequency", 10, type=int)
        warnings_list = (context.setdefault('processing_warnings', [])
                         if context is not None else [])

        fx = data[:, self.peak_position]

        # Normalise prominence against signal dynamic range
        signal_range   = float(np.percentile(fx, self.SIGNAL_PCT_HIGH) -
                               np.percentile(fx, self.SIGNAL_PCT_LOW))
        effective_prom = max(self.min_height_na, self.prominence_fraction * signal_range)

        # Initial candidate detection
        peaks, _ = find_peaks(
            fx,
            prominence=effective_prom,
            distance=max(self.MIN_PEAK_WIDTH_SCANS, int(self.MIN_PEAK_DISTANCE_SEC * freq)),
            height=self.min_height_na,
            width=self.MIN_PEAK_WIDTH_SCANS,
        )

        if len(peaks) == 0:
            warnings_list.append(
                f"Peak detector: no candidates found (prominence threshold "
                f"{effective_prom:.3f} nA = {self.prominence_fraction * 100:.0f}% of signal "
                f"range). Using signal maximum as fallback."
            )
            _pos = int(np.argmax(fx))
            if context is not None:
                context['peak_amplitude_positions'] = _pos
                context['peak_amplitude_values']    = float(fx[_pos])
            return data

        # Validate candidates with adaptive rise/decay windows
        valid_peaks, valid_vals, valid_meta = [], [], []
        for idx in peaks:
            ok, info = self._validate_peak(fx, idx, freq)
            if ok:
                valid_peaks.append(idx)
                valid_vals.append(float(fx[idx]))
                valid_meta.append(info)

        if len(valid_vals) == 0:
            warnings_list.append(
                f"Peak detector: {len(peaks)} candidate(s) found but none passed "
                f"rise/decay validation. Using signal maximum as fallback."
            )
            _pos = int(np.argmax(fx))
            if context is not None:
                context['peak_amplitude_positions'] = _pos
                context['peak_amplitude_values']    = float(fx[_pos])
            return data

        # Select highest-amplitude validated peak
        best     = int(np.argmax(valid_vals))
        peak_pos = int(valid_peaks[best])
        peak_val = float(valid_vals[best])
        sel_meta = valid_meta[best]

        # Write full context
        if context is not None:
            context['peak_amplitude_positions'] = peak_pos
            context['peak_amplitude_values']    = peak_val

            rs = sel_meta['rise_window_samples']
            ds = sel_meta['decay_window_samples']

            context['decay_left_region'] = {
                'indices':         list(range(max(0, peak_pos - rs), peak_pos)),
                'values':          fx[max(0, peak_pos - rs): peak_pos].tolist(),
                'time_window_sec': sel_meta['rise_time_sec'],
                'adaptive':        True,
            }
            right_end = min(len(fx), peak_pos + ds + 1)
            context['decay_right_region'] = {
                'indices':         list(range(peak_pos + 1, right_end)),
                'values':          fx[peak_pos + 1: right_end].tolist(),
                'time_window_sec': sel_meta['decay_time_sec'],
                'adaptive':        True,
            }
            context['decay_validation_params'] = {
                'adaptive_windows':       True,
                'rise_window_sec':        sel_meta['rise_time_sec'],
                'decay_window_sec':       sel_meta['decay_time_sec'],
                'rise_window_samples':    rs,
                'decay_window_samples':   ds,
                'acquisition_frequency':  freq,
                'peak_passed_validation': True,
                'validation_type':        'adaptive_neurotransmitter_profile',
            }

        return data
