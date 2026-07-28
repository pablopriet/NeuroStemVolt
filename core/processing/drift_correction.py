import numpy as np
from .base import Processor


class DriftCorrection(Processor):
    """
    Corrects for electrode drift in FSCV color-plot data.

    For each voltage step (horizontal line in the color plot), the mean current
    across each file is tracked over the baseline period. A linear trend is fitted
    to how that mean changes across baseline files, then the extrapolated trend
    value is subtracted from the corresponding voltage step column in every file.

    This operates on the 2D data matrix directly, so the color plot and I-T trace
    both reflect the corrected signal.

    Requires cross-file context (same mechanism as Normalize) — the processor
    accumulates baseline means across files and fits the trend once all baseline
    files have been processed.

    Args:
        files_before_treatment (int or None): Number of baseline files used to fit
            the drift trend. Requires ≥ 2. Reads from QSettings if not provided.
    """

    def __init__(self, files_before_treatment=None):
        self.files_before_treatment = files_before_treatment

    def process(self, data, context=None):
        if context is None:
            return data

        # Resolve baseline file count
        n_baseline = self.files_before_treatment
        if n_baseline is None:
            from PyQt5.QtCore import QSettings
            n_baseline = QSettings("HashemiLab", "NeuroStemVolt").value(
                "files_before_treatment", 0, type=int)

        if n_baseline < 2:
            return data  # need at least 2 baseline files to fit a trend

        n_timepoints, n_voltages = data.shape

        # Initialise shared state on first file
        if "drift_file_index" not in context:
            context["drift_file_index"] = 0
            # Accumulates shape: (n_baseline_files, n_voltages)
            context["drift_baseline_means"] = []
            # Fitted trend per voltage step: shape (n_voltages,) each for slope & intercept
            context["drift_slopes"] = None
            context["drift_intercepts"] = None

        file_idx = context["drift_file_index"]

        if file_idx < n_baseline:
            # Baseline phase
            # For each voltage step, average across time in this file
            # data shape: (time_points, voltage_steps)
            col_means = np.mean(data, axis=0)  # shape: (n_voltages,)
            context["drift_baseline_means"].append(col_means)

            # After the last baseline file, fit a linear trend per voltage step
            if file_idx == n_baseline - 1:
                baseline_matrix = np.array(context["drift_baseline_means"])
                # shape: (n_baseline, n_voltages)
                file_indices = np.arange(n_baseline, dtype=float)
                slopes = np.zeros(n_voltages)
                intercepts = np.zeros(n_voltages)
                for v in range(n_voltages):
                    coeffs = np.polyfit(file_indices, baseline_matrix[:, v], 1)
                    slopes[v] = coeffs[0]
                    intercepts[v] = coeffs[1]
                context["drift_slopes"] = slopes
                context["drift_intercepts"] = intercepts
                print(f"DriftCorrection: fitted trend across {n_baseline} baseline files "
                      f"for {n_voltages} voltage steps.")

            corrected = data  # baseline files returned unchanged

        else:
            # Stimulation phase
            slopes = context.get("drift_slopes")
            intercepts = context.get("drift_intercepts")

            if slopes is not None and intercepts is not None:
                # Anchor correction to the last baseline file so that file has
                # zero correction — only the *change* beyond baseline is removed.
                # correction(t) = slope * (t - (n_baseline - 1))
                drift_ref = slopes * (n_baseline - 1) + intercepts
                drift = slopes * file_idx + intercepts
                correction = drift - drift_ref  # shape: (n_voltages,)
                corrected = data - correction[np.newaxis, :]  # broadcast over time axis
            else:
                corrected = data  # trend not available — return unchanged

        context["drift_file_index"] += 1
        return corrected
