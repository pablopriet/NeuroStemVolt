from .base import Processor
import numpy as np
from scipy.signal import find_peaks
from PyQt5.QtCore import QSettings


class FindAmplitudeMultiple(Processor):
    """
    Detect multiple spontaneous peaks in the I-T profile.

    If no valid peaks are found, context is updated with empty lists and a
    warning is written to ``context['processing_warnings']``.

    Args:
        peak_position (int): Column index of the voltage step to analyse.
        prominence_fraction (float): Minimum prominence as a fraction of
            signal range (default 0.05 = 5 %).
        min_height_na (float): Absolute nA floor (default 0.02 nA).
        max_peaks (int): Maximum number of peaks to keep (default 10).
        min_peak_distance_sec (float | None): Minimum gap between peaks in
            seconds.  Defaults to the class constant MIN_PEAK_DISTANCE_SEC.
    """

    # ── Detection constants ──────────────────────────────────────────────────
    DEFAULT_PROMINENCE_FRACTION = 0.05
    DEFAULT_MIN_HEIGHT_NA       = 0.02
    MIN_PEAK_DISTANCE_SEC       = 0.5   # minimum gap between candidate peaks (s)
    MIN_PEAK_WIDTH_SCANS        = 2     # minimum scan width — rejects single-scan spikes

    # ── Adaptive validation window limits ────────────────────────────────────
    MIN_RISE_TIME_SEC  = 0.2
    MAX_RISE_TIME_SEC  = 3.0
    MIN_DECAY_TIME_SEC = 1.0
    MAX_DECAY_TIME_SEC = 10.0

    # ── Signal-range percentiles for normalisation ───────────────────────────
    SIGNAL_PCT_LOW  = 5
    SIGNAL_PCT_HIGH = 95

    def __init__(self, peak_position, prominence_fraction=0.05, min_height_na=0.02,
                 rise_window_sec=1.0, decay_window_sec=20.0, max_peaks=10,
                 min_peak_distance_sec=None):
        self.peak_position         = peak_position
        self.prominence_fraction   = prominence_fraction
        self.min_height_na         = min_height_na
        self.rise_window_sec       = rise_window_sec
        self.decay_window_sec      = decay_window_sec
        self.max_peaks             = max_peaks
        self.min_peak_distance_sec = (min_peak_distance_sec
                                      if min_peak_distance_sec is not None
                                      else self.MIN_PEAK_DISTANCE_SEC)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _find_adaptive_time_windows(self, fx, peak_idx, freq):
        min_rise  = max(3, int(self.MIN_RISE_TIME_SEC  * freq))
        max_rise  =        int(self.MAX_RISE_TIME_SEC  * freq)
        min_decay = max(5, int(self.MIN_DECAY_TIME_SEC * freq))
        max_decay =        int(self.MAX_DECAY_TIME_SEC * freq)

        rise_w  = self._find_rise_window(fx, peak_idx, min_rise, max_rise)
        decay_w = self._find_decay_window(fx, peak_idx, fx[peak_idx], min_decay, max_decay)

        return rise_w, decay_w, rise_w / freq, decay_w / freq

    def _find_rise_window(self, fx, peak_idx, min_samples, max_samples):
        peak_val     = fx[peak_idx]
        start_search = max(0, peak_idx - max_samples)
        early        = fx[start_search: max(1, start_search + min_samples // 2)]
        baseline     = float(np.median(early)) if len(early) > 0 else 0.0
        # More sensitive threshold for spontaneous signals
        threshold    = baseline + 0.05 * (peak_val - baseline)

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

        far_start  = min(len(fx) - 5, peak_idx + max_samples - 5)
        far_points = fx[far_start: len(fx)]
        baseline   = float(np.median(far_points)) if len(far_points) > 2 else (
                     float(fx[-1]) if len(fx) > 0 else 0.0)

        # Adjusted thresholds for spontaneous events
        t70 = baseline + 0.7 * (peak_val - baseline)
        t30 = baseline + 0.3 * (peak_val - baseline)
        idx70 = idx30 = None

        for i in range(peak_idx + 1, end_search):
            if i >= len(fx):
                break
            if idx70 is None and fx[i] <= t70:
                idx70 = i
            if idx30 is None and fx[i] <= t30:
                idx30 = i
                break

        if idx30 is not None:
            w = min(max_samples, (idx30 - peak_idx) + 5)
        elif idx70 is not None:
            w = min(max_samples, (idx70 - peak_idx) * 2)
        else:
            w = self._find_decay_by_slope(fx, peak_idx, min_samples, max_samples)

        return max(min_samples, w)

    def _find_decay_by_slope(self, fx, peak_idx, min_samples, max_samples):
        end_search  = min(len(fx), peak_idx + max_samples)
        window_size = 3  # smaller window for spontaneous signals
        for i in range(peak_idx + window_size, end_search - window_size):
            if i + window_size >= len(fx):
                break
            slope = (fx[i + window_size] - fx[i - window_size]) / (2 * window_size)
            if abs(slope) < 0.002:
                return max(min_samples, min(max_samples, i - peak_idx + 5))
        return min(max_samples, min_samples * 2)

    def _validate_peak(self, fx, peak_idx, freq):
        """Validate a spontaneous candidate using adaptive rise/decay windows."""
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
        if end_idx <= peak_idx + 3:
            return False
        region = fx[peak_idx: end_idx]
        if len(region) < 3:
            return False
        try:
            slope = np.polyfit(np.arange(len(region)), region, 1)[0]
            if slope >= 0:
                return False
            end_val = float(np.median(region[-3:]))
            # More lenient decay requirement for spontaneous signals
            return (peak_val - end_val) / peak_val >= 0.1 if peak_val != 0 else False
        except Exception:
            return False

    def _check_rise(self, fx, peak_idx, rise_window):
        peak_val  = fx[peak_idx]
        start_idx = max(0, peak_idx - rise_window)
        if peak_idx - start_idx < 3:
            return False
        region = fx[start_idx: peak_idx + 1]
        try:
            slope = np.polyfit(np.arange(len(region)), region, 1)[0]
            if slope <= 0:
                return False
            # More lenient rise requirement for spontaneous signals
            return float(peak_val - np.median(region[:3])) >= 0.03
        except Exception:
            return False

    # ── Public interface ─────────────────────────────────────────────────────

    def process(self, data, context=None):
        """
        Detect multiple spontaneous peaks and write metadata to context.
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
            distance=max(self.MIN_PEAK_WIDTH_SCANS,
                         int(self.min_peak_distance_sec * freq)),
            height=self.min_height_na,
            width=self.MIN_PEAK_WIDTH_SCANS,
        )

        # Validate all candidates
        valid_peaks, valid_vals, valid_meta = [], [], []
        for idx in peaks:
            ok, info = self._validate_peak(fx, idx, freq)
            if ok:
                valid_peaks.append(idx)
                valid_vals.append(float(fx[idx]))
                valid_meta.append(info)

        if len(valid_peaks) == 0:
            if len(peaks) > 0:
                warnings_list.append(
                    f"Multi-peak detector: {len(peaks)} candidate(s) found but none passed "
                    f"rise/decay validation (prominence threshold {effective_prom:.3f} nA). "
                    f"No peaks stored for this file."
                )
            else:
                warnings_list.append(
                    f"Multi-peak detector: no candidates found (prominence threshold "
                    f"{effective_prom:.3f} nA = {self.prominence_fraction * 100:.0f}% of signal range)."
                )
            if context is not None:
                context['peak_amplitude_positions'] = []
                context['peak_amplitude_values']    = []
                context['num_peaks_detected']       = 0
            return data

        # Keep top-N by amplitude, then sort chronologically
        sorted_by_amp = np.argsort(valid_vals)[::-1]
        keep_n        = min(self.max_peaks, len(valid_peaks))
        top_i         = sorted_by_amp[:keep_n]
        time_order    = np.argsort([valid_peaks[i] for i in top_i])

        final_peaks = [valid_peaks[top_i[i]] for i in time_order]
        final_vals  = [valid_vals [top_i[i]] for i in time_order]
        final_meta  = [valid_meta [top_i[i]] for i in time_order]

        # Write context
        if context is not None:
            context['peak_amplitude_positions'] = final_peaks
            context['peak_amplitude_values']    = final_vals
            context['num_peaks_detected']       = len(final_peaks)
            context['all_peak_metadata']        = final_meta

            # Primary peak = highest amplitude (for decay-fitting downstream)
            max_amp_i    = int(np.argmax(final_vals))
            primary_pos  = final_peaks[max_amp_i]
            primary_meta = final_meta[max_amp_i]

            context['primary_peak_position'] = primary_pos
            context['primary_peak_value']    = final_vals[max_amp_i]

            rs = primary_meta['rise_window_samples']
            ds = primary_meta['decay_window_samples']

            context['decay_left_region'] = {
                'indices':         list(range(max(0, primary_pos - rs), primary_pos)),
                'values':          fx[max(0, primary_pos - rs): primary_pos].tolist(),
                'time_window_sec': primary_meta['rise_time_sec'],
                'adaptive':        True,
            }
            right_end = min(len(fx), primary_pos + ds + 1)
            context['decay_right_region'] = {
                'indices':         list(range(primary_pos + 1, right_end)),
                'values':          fx[primary_pos + 1: right_end].tolist(),
                'time_window_sec': primary_meta['decay_time_sec'],
                'adaptive':        True,
            }
            context['decay_validation_params'] = {
                'adaptive_windows':       True,
                'rise_window_sec':        primary_meta['rise_time_sec'],
                'decay_window_sec':       primary_meta['decay_time_sec'],
                'rise_window_samples':    rs,
                'decay_window_samples':   ds,
                'acquisition_frequency':  freq,
                'peak_passed_validation': True,
                'validation_type':        'adaptive_spontaneous_multiple_peaks',
                'signal_type':            'spontaneous',
                'multiple_peaks':         True,
            }

        return data
