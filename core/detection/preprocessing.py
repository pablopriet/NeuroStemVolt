# python
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


class Preprocessing:


    def __init__(self):
        pass

    def estimate_noise_sigma(
            x: np.ndarray | Sequence[float],
            *,
            # High-pass options (drift removal)
            sample_rate_hz: Optional[float] = None,
            highpass_hz: Optional[float] = None,
            highpass_window_sec: Optional[float] = None,
            # Peak exclusion
            peak_idx: Optional[Iterable[int]] = None,
            peak_exclusion_sec: Optional[float] = None,
            peak_exclusion_samples: Optional[int] = None,
            # Robustness
            robust_center: bool = True,
            # Diagnostics
            return_details: bool = False,
    ) -> float | Tuple[float, dict]:
        """
        Estimate noise standard deviation using MAD with optional high-pass to reduce drift.

        Parameters
        ----------
        x : 1D array-like
            Input signal.
        sample_rate_hz : float, optional
            Sampling rate. Required if using highpass_hz or highpass_window_sec, or when
            specifying peak_exclusion_sec.
        highpass_hz : float, optional
            If provided with sample_rate_hz, applies a simple one-pole high-pass filter
            with the given cutoff frequency in Hz to reduce slow drift.
        highpass_window_sec : float, optional
            If provided with sample_rate_hz (and highpass_hz is not given), subtracts a
            moving-average baseline with this window length (seconds), acting as a
            simple high-pass/detrend.
        peak_idx : iterable of int, optional
            Indices of detected peaks to exclude from noise estimation.
        peak_exclusion_sec : float, optional
            Half-width (seconds) to exclude around each peak index. Requires sample_rate_hz.
        peak_exclusion_samples : int, optional
            Half-width (samples) to exclude around each peak index. If both exclusion in
            seconds and samples are provided, samples takes precedence.
        robust_center : bool, default True
            If True, center data using the median before MAD. If False, use the mean.
        return_details : bool, default False
            If True, returns (sigma, details_dict).

        Returns
        -------
        sigma : float
            Estimated noise standard deviation.
        details : dict (optional)
            Contains intermediate info: {'mad': float, 'n_used': int, 'center': float,
            'method': str, 'detrend': str}
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size

        if n == 0:
            return (0.0, {"mad": 0.0, "n_used": 0, "center": math.nan, "method": "MAD",
                          "detrend": "none"}) if return_details else 0.0

        # Ensure finites
        finite_mask = np.isfinite(x)
        x = x[finite_mask]
        if x.size == 0:
            return (0.0, {"mad": 0.0, "n_used": 0, "center": math.nan, "method": "MAD",
                          "detrend": "none"}) if return_details else 0.0

        detrend_used = "none"

        # Optional high-pass detrend using one of two strategies
        if highpass_hz is not None:
            if not sample_rate_hz:
                raise ValueError("sample_rate_hz must be provided when using highpass_hz.")
            if highpass_hz <= 0 or highpass_hz >= sample_rate_hz / 2:
                # Degenerate or invalid cutoff; skip high-pass
                pass
            else:
                # One-pole RC high-pass: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
                dt = 1.0 / sample_rate_hz
                rc = 1.0 / (2.0 * math.pi * highpass_hz)
                alpha = rc / (rc + dt)
                y = np.empty_like(x)
                y[0] = 0.0
                # Use explicit loop (fast enough for typical lengths)
                for i in range(1, x.size):
                    y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
                x = y
                detrend_used = f"one_pole_hp(fc={highpass_hz:g}Hz)"
        elif highpass_window_sec is not None:
            if not sample_rate_hz:
                raise ValueError("sample_rate_hz must be provided when using highpass_window_sec.")
            win = int(round(highpass_window_sec * sample_rate_hz))
            # Enforce a reasonable odd window length
            win = max(3, win)
            if win % 2 == 0:
                win += 1
            # Moving-average baseline subtraction (acts as a simple high-pass)
            kernel = np.ones(win, dtype=float) / win
            baseline = np.convolve(x, kernel, mode="same")
            x = x - baseline
            detrend_used = f"moving_avg_sub(window={highpass_window_sec:g}s)"

        # Build mask to exclude regions around known peaks, if provided
        mask = np.ones(x.size, dtype=bool)
        if peak_idx is not None:
            # Compute exclusion half-width in samples
            if peak_exclusion_samples is not None:
                r = int(peak_exclusion_samples)
            elif peak_exclusion_sec is not None:
                if not sample_rate_hz:
                    raise ValueError("sample_rate_hz must be provided when using peak_exclusion_sec.")
                r = int(round(peak_exclusion_sec * sample_rate_hz))
            else:
                r = 0

            r = max(0, r)

            if r > 0:
                # Note: peak indices were defined in the original (pre-finite) domain.
                # We excluded non-finite samples earlier; we cannot directly map indices.
                # If you expect NaNs, prefer providing a boolean mask aligned with 'x' after cleaning.
                # Here, we assume peak_idx refers to the current 'x' indexing (all finite).
                for p in peak_idx:
                    if p is None:
                        continue
                    i = int(p)
                    start = max(0, i - r)
                    stop = min(x.size, i + r + 1)
                    mask[start:stop] = False

        # Select samples to use
        used = x[mask]
        n_used = used.size

        # If exclusion leaves too few samples, fall back to using all
        if n_used < 3:
            used = x
            n_used = used.size

        if n_used == 0:
            return (0.0, {"mad": 0.0, "n_used": 0, "center": math.nan, "method": "MAD",
                          "detrend": detrend_used}) if return_details else 0.0

        # Robust center
        center = np.median(used) if robust_center else float(np.mean(used))

        # MAD and sigma conversion for Gaussian noise
        mad = float(np.median(np.abs(used - center)))
        # 1 / Phi^{-1}(3/4) ≈ 1.4826
        c = 1.482653
        sigma = c * mad

        # Guard tiny negative due to float noise
        if sigma < 0:
            sigma = 0.0

        if return_details:
            return sigma, {
                "mad": mad,
                "n_used": int(n_used),
                "center": float(center),
                "method": "MAD",
                "detrend": detrend_used,
            }
        return sigma

