from .base import Processor
import numpy as np

class BaselineCorrection(Processor):
    """
    Subtracts baseline current from FSCV data using a pre-stimulation / pre-injection time window.
    """

    def process(self, data, context=None):
        """
        Apply baseline correction to the data using the pre-stimulation or pre-injection region.

        Args:
            data (np.ndarray): 2D FSCV array (voltage × time).
            context (dict): Must contain either 'stim_start' or 'injection_start'.
                            May include 'peak_position' and 'acquisition_frequency'.

        Returns:
            np.ndarray: Baseline-corrected data.
        """
        if context is None:
            raise ValueError("Context is required for BaselineCorrection.")

        has_stim = "stim_start" in context
        has_inj  = "injection_start" in context

        if not has_stim and not has_inj:
            raise ValueError(
                "BaselineCorrection requires either 'stim_start' (stimulation) "
                "or 'injection_start' (flow cell) in the context."
            )

        # Prefer stimulation start if available, otherwise use injection start
        stim_or_inj_start = context.get("stim_start", None)
        if stim_or_inj_start is None:
            stim_or_inj_start = context.get("injection_start", 0.0)

        peak_position = context.get("peak_position", 257)
        acquisition_frequency = context.get("acquisition_frequency", 10)

        # convert time (s) to sample index
        stim_start_idx = int(stim_or_inj_start * acquisition_frequency)

        # IT profile along the peak-position voltage
        # data: 2D (voltage × time), so we index rows by voltage, columns by time
        it_trace = data[peak_position, :]  # shape (time,)
        baseline_region = it_trace[:stim_start_idx] if stim_start_idx > 0 else it_trace

        if baseline_region.size == 0:
            baseline = 0.0
        else:
            baseline = float(np.mean(baseline_region))

        # store scalar baseline in context
        context["baseline"] = baseline

        # subtract from all voltages at each time point
        data = data - baseline

        return data
