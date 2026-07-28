from core.spheroid_experiment import SpheroidExperiment
from core.processing.exponentialdecay import exp_decay
from core.processing.normalize import Normalize
import os
import numpy as np

LN2 = np.log(2)


def _model_AkC(t, A, k, C):
    # Re-zeroed at each replicate's own peak: A is the value AT the peak (t=0), NOT amplitude above baseline.
    return (A - C) * np.exp(-k * t) + C


def draw_exponential_fit(ax, result, freq):
    """Draw a joint exponential fit onto ``ax``, in seconds.

    This is the SINGLE implementation shared by the on-screen canvas
    (``PlotCanvas.show_decay_exponential_fitting``) and the exported figure
    (``GroupAnalysis.plot_exponential_fit_aligned``). They used to be two
    separate bodies that had drifted apart: the export drew a points x-axis
    with a mean +/- SD band, while the UI drew a seconds x-axis with the raw
    data scattered, so the exported plot did not look like the one on screen.

    Args:
        ax (matplotlib.axes.Axes): axes to draw into.
        result (dict): output of ``GroupAnalysis.exponential_fitting_joint``.
        freq (float): acquisition frequency in Hz, used to convert the fit's
            sample units to seconds.

    Returns:
        matplotlib.axes.Axes: the axes that were drawn into.
    """
    from scipy.stats import t as t_dist

    cropped_ITs = result["cropped_ITs"]
    time_all = result["time_all"]
    A_fit, k_fit, C_fit = result["A"], result["k"], result["C"]
    t_half = result["t_half"]

    n_exps, n_post = cropped_ITs.shape
    t_rel = np.arange(n_post) / freq                 # seconds

    # Fit is evaluated in SAMPLE space (k is per sample), then mapped to seconds.
    t_fit_pts = np.linspace(0, n_post - 1, 500)
    y_fit = _model_AkC(t_fit_pts, A_fit, k_fit, C_fit)
    t_fit_rel = t_fit_pts / freq

    # 95% CI of the fit via the Jacobian and the full 3x3 covariance.
    tval = t_dist.ppf(0.975, max(0, result["nu"]))
    J = np.empty((len(t_fit_pts), 3))
    J[:, 0] = np.exp(-t_fit_pts * k_fit)                                  # d/dA
    J[:, 1] = -(A_fit - C_fit) * t_fit_pts * np.exp(-t_fit_pts * k_fit)   # d/dk
    J[:, 2] = 1 - np.exp(-t_fit_pts * k_fit)                              # d/dC
    pcov = np.asarray(result["pcov"], dtype=float)
    ci = np.sqrt(np.sum((J @ pcov) * J, axis=1)) * tval

    # a) each replicate in light grey
    for row in cropped_ITs:
        ax.plot(t_rel, row, color='gray', alpha=0.3, lw=1, label='_nolegend_')

    # b) individual data points used for fitting
    ax.scatter(np.asarray(time_all) / freq, cropped_ITs.flatten(),
               color='black', s=16, alpha=0.7, label='Data points')

    # c) fitted exponential curve and its 95% CI
    ax.plot(t_fit_rel, y_fit, color='C1', lw=2, label='Exp fit')
    ax.fill_between(t_fit_rel, y_fit - ci, y_fit + ci, color='C1', alpha=0.3, label='95% CI')

    # d) half-life marker
    t_half_s = t_half / freq
    ax.axvline(t_half_s, color='magenta', ls='--', label=f't½ ≈ {t_half_s:.2f} s')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Current (nA)', fontsize=12)
    ax.set_title('Post-peak IT decays & exponential fit', fontsize=14)
    ax.legend(frameon=False)
    ax.grid(False)

    max_t = t_rel[-1]
    tick_interval = 5  # seconds
    ax.set_xticks(np.arange(0, max_t + tick_interval, tick_interval))
    return ax

class GroupAnalysis:
    """
    Initializes a GroupAnalysis instance with optional experiments.
        Args:
            experiments (None, SpheroidExperiment, or list): Optional initial experiments to add.
        Raises:
            ValueError: If input is not None, a SpheroidExperiment, or a list of SpheroidExperiment instances.
        """
    def __init__(self, experiments=None):
        if experiments is None:
            self.experiments = []
        elif isinstance(experiments, SpheroidExperiment):
            self.experiments = [experiments]
        elif isinstance(experiments, list):
            self.experiments = experiments
        else:
            raise ValueError("experiments must be None, a SpheroidExperiment, or a list of SpheroidExperiment objects.")
    
    @staticmethod
    def _resolve_peak_position(metadata):
        """Return a single integer peak position from metadata.

        For single-peak files the position is already a scalar.
        For multi-peak files it is a list — we return the active peak
        (stored as 'peak_amplitude_active_index') or fall back to the first one.
        """
        pos = metadata.get("peak_amplitude_positions")
        if pos is None:
            return 0
        if isinstance(pos, (list, np.ndarray)):
            if len(pos) == 0:
                return 0
            active = int(metadata.get("peak_amplitude_active_index", 0))
            active = max(0, min(active, len(pos) - 1))
            return int(pos[active])
        try:
            return int(pos)
        except (TypeError, ValueError):
            return 0

    def add_experiment(self, *experiments):
        """
        Add one or more SpheroidExperiment instances to the group analysis.
        Args:
            experiments: One or more SpheroidExperiment instances to be added.
        """
        for experiment in experiments:
            if isinstance(experiment, SpheroidExperiment):
                self.experiments.append(experiment)
            elif isinstance(experiment, list):
                for exp in experiment:
                    if isinstance(exp, SpheroidExperiment):
                        self.experiments.append(exp)
                    else:
                        raise ValueError("All elements in the list must be SpheroidExperiment instances.")
            else:
                raise ValueError("Arguments must be SpheroidExperiment instances or lists of them.")

    def get_single_experiments(self, index: int):
        """
        Retrieves a single experiment by index.
        Args:
            index (int): Index of the experiment to retrieve.
        Returns:
            SpheroidExperiment: The selected experiment.
        """
        return self.experiments[index]
    
    def get_experiments(self):
        """
        Returns all experiments.
        Returns:
            list: List of SpheroidExperiment instances.
        """
        return self.experiments
    
    def clear_experiments(self):
        """Remove all experiments from this group."""
        self.experiments.clear()
        #print("Replicates Cleared")
        #print(self.get_experiments())

    def clear_single_experiment(self, index):
        """Remove single experiment from group analysis"""
        del self.experiments[index]

    def get_timepoints_minutes(self):
        """Experiment time axis in minutes, one entry per file.

        Baseline files (those recorded before the treatment) are given NEGATIVE
        times so that the first file recorded after the treatment is time zero:

            time_min = (file_index - files_before_treatment) * time_between_files

        This is the same convention the CSV exports use, so the plots and the
        exported tables share one time axis. When no baseline files are
        configured the axis simply starts at zero.

        Returns:
            np.ndarray: time in minutes for each file index.
        """
        if not self.experiments:
            return np.array([])
        exp = self.experiments[0]
        n_files = exp.get_file_count()
        n_before = exp.get_number_of_files_before_treatment() or 0
        interval = exp.get_time_between_files()
        return np.array([(i - n_before) * interval for i in range(n_files)], dtype=float)

    @staticmethod
    def _time_ticks(time_points, step=None):
        """X ticks covering a (possibly negative) time axis, anchored on zero.

        The old plots hard-coded ``np.arange(0, max+1, 10)``, which silently
        clipped every baseline point once the axis was allowed to go negative.
        Anchoring on zero keeps the treatment start on a tick.

        Args:
            time_points (array-like): the time axis being plotted, in minutes.
            step (float, optional): spacing between ticks, in minutes. When
                omitted, a readable spacing is chosen from the span so labels
                do not collide.

        Returns:
            np.ndarray: tick positions spanning the whole axis.
        """
        time_points = np.asarray(time_points, dtype=float)
        if time_points.size == 0:
            return np.array([0.0])
        if step is None:
            step = GroupAnalysis._nice_time_step(time_points)
        lo, hi = float(np.min(time_points)), float(np.max(time_points))
        first = np.floor(lo / step) * step
        return np.arange(first, hi + step, step)

    # Tick spacings (minutes) worth labelling — no finer than 5 min, so a dense
    # axis reads 0, 5, 10 … rather than every individual timepoint.
    NICE_TICK_STEPS = (5, 10, 15, 20, 30, 60, 120, 180, 300, 600)

    @staticmethod
    def _nice_time_step(time_points, max_ticks=10):
        """Smallest readable tick spacing that keeps the axis under max_ticks.

        Args:
            time_points (array-like): the time axis being plotted, in minutes.
            max_ticks (int): rough upper bound on how many ticks to draw.

        Returns:
            float: spacing in minutes, taken from NICE_TICK_STEPS.
        """
        time_points = np.asarray(time_points, dtype=float)
        if time_points.size == 0:
            return GroupAnalysis.NICE_TICK_STEPS[0]
        span = float(np.max(time_points) - np.min(time_points))
        for step in GroupAnalysis.NICE_TICK_STEPS:
            if span / step <= max_ticks:
                return step
        return GroupAnalysis.NICE_TICK_STEPS[-1]

    def set_processing_options_exp(self, processors = None):
        """Set the data processing pipeline for all experiments.

        Applies a shared list of processing steps (e.g., normalization, filtering)
        to all experiments in the group.

        Args:
            processors (list): A list of processor instances to apply.
        """
        for exp in self.experiments:
            exp.set_processing_steps(processors)

    def non_normalized_first_ITs(self):
        """Get unprocessed first I-T signals from each replicate.

        This function extracts the initial stimulation file from each experiment
        and returns the raw current traces (I-T curves) prior to any processing.

        Returns:
            np.ndarray: A matrix (n_experiments x n_timepoints) of raw IT traces.
        """
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None
        
        # Assume all experiments have the same number of files/timepoints
        file_count = self.experiments[0].get_file_count()
        n_timepoints = self.experiments[0].get_file_time_points()

        ITs = np.empty((n_experiments, n_timepoints), dtype=float)

        for i, experiment in enumerate(self.experiments):
            first_file = experiment.get_spheroid_file(0)
            IT_individual = first_file.get_original_data_IT()
            ITs[i, :] = IT_individual
        
        return ITs
    
    def get_all_reuptake_curves(self):
        """Retrieves and aligns IT signals post-peak for all replicates.
        Returns:
            np.ndarray: Aligned post-peak IT data across all replicates.
        """
        from scipy.optimize import curve_fit
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None
        # Assume all experiments have the same number of files/timepoints
        n_timepoints = self.experiments[0].get_file_time_points()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment() # This will be zero if no files before treatment
        file_count = self.experiments[0].get_file_count()

        all_ITs = np.empty((n_experiments*file_count, n_timepoints))
        peak_amplitude_positions = []
        
        for i, experiment in enumerate(self.experiments):
            for j, spheroid_file in enumerate(experiment.files):
                spheroid_file = experiment.get_spheroid_file(j)
                IT_individual = spheroid_file.get_processed_data_IT()
                metadata = spheroid_file.get_metadata()
                peak_amplitude_positions.append(self._resolve_peak_position(metadata))
                all_ITs[i*file_count+j, :] = IT_individual

        peaks = [int(p) for p in peak_amplitude_positions]
        min_peak = np.min(peaks)

        pre_allocated_ITs_array = np.full((n_experiments*file_count, n_timepoints - min_peak), np.nan)       
        # Fill the pre-allocated array with the cropped ITs, starting from the peak position
        for i, (row, peak) in enumerate(zip(all_ITs, peaks)):
            print(i)
            peak = int(peak)
            cropped = row[peak:]
            length = cropped.shape[0]
            pre_allocated_ITs_array[i, :length] = cropped
            
        print(pre_allocated_ITs_array)
        return pre_allocated_ITs_array

    def average_IT_over_replicates(self):
        """Compute the average IT curve over replicates for each timepoint.

        Returns:
            np.ndarray: Averaged IT matrix with shape (n_files, n_timepoints),
            where each row corresponds to a timepoint and each column to a file.
        """
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None

        # Assume all experiments have the same number of files/timepoints
        file_count = self.experiments[0].get_file_count()
        n_timepoints = self.experiments[0].get_file_time_points()

        all_ITs = np.empty((file_count, n_timepoints, n_experiments), dtype=float) 
        # 16 x 600 x 4 (n_experiments x n_timepoints x n_replicates)
        # Then do the average over the replicates
        for i, experiment in enumerate(self.experiments):
            for j, spheroid_file in enumerate(experiment.files):
                #print(spheroid_file.get_filepath())
                IT_individual = spheroid_file.get_processed_data_IT()
                all_ITs[j, :, i] = IT_individual
        # Average over the third dimension (replicates)
        mean_ITs = np.nanmean(all_ITs, axis=2)
        print(np.shape(mean_ITs))
        
        return mean_ITs
    
    def amplitudes_first_stim(self):
        """Retrieve unnormalized amplitudes from the first stimulation file of each replicate.

        This pulls metadata from the first stimulation and returns raw amplitudes.
        Normalization should not have been applied to these experiments.

        Returns:
            list of list: Nested list containing peak amplitude values for each replicate.

        Raises:
            RuntimeError: If Normalize is found in any experiment's processor pipeline.
        """
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None

        # Assume all experiments have the same number of files/timepoints
        n_timepoints = self.experiments[0].get_file_count()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment() # This will be zero if no files before treatment
        time_points = self.get_timepoints_minutes()
        amplitudes = []
        for i, experiment in enumerate(self.experiments):
            first_stim_spheroid = experiment.get_spheroid_file(0) #Getting the first stimulation
            # print(first_stim_spheroid.get_filepath()) 
            # Check for Normalize in the processor list
            has_norm = any(isinstance(p, Normalize) for p in experiment.processors)
            if has_norm:
                raise RuntimeError("Experiment must not include Normalize()")
            metadata = first_stim_spheroid.get_metadata()
            peak_amplitude_values = metadata['peak_amplitude_values']
            amplitudes.append(peak_amplitude_values.tolist())

        return amplitudes

    def amplitudes_over_time_single_experiment(self, experiment_index=0):
        """Retrieve amplitude data over time from a single experiment.
        Args:
            experiment_index (int): Index of the experiment to inspect.
        Returns:
            tuple: (time_points, amplitudes, files_before_treatment)
        """
        experiment = self.get_single_experiments(experiment_index)
        files_before_treatment = experiment.get_number_of_files_before_treatment()
        amplitudes = []
        time_points = self.get_timepoints_minutes()
        for spheroid_file in experiment.files:
            # Current metadata:
            # dict_keys(['peak_position', 'stim_start', 'stim_duration', 'stim_frequency', 
            # 'background_subtraction_region', 'baseline', 'experiment_first_peak', 'peak_amplitude_positions', 
            # 'peak_amplitude_values', 'exponential fitting parameters']) dict_values([257, 5.0, 2.0, 20, (0, 10), 
            # array([0.02988434]), np.float64(0.3822367999525832), array([74]), array([1.04633804]), 
            # {'A': np.float64(1.5705880537206165), 'tau': np.float64(119.00028560994886), 
            # 'C': np.float64(0.22708457061563378), 't_half': np.float64(82.48471245636428)}])
            metadata = spheroid_file.get_metadata()
            peak_amplitude_values = metadata['peak_amplitude_values']
            amplitudes.append(peak_amplitude_values.tolist())
        print(f"Time points: {time_points}")
        print(f"Amplitudes: {amplitudes}")
        return time_points, amplitudes, files_before_treatment
    
    def amplitudes_over_time_all_experiments(self):
        """Compute mean amplitude over time across all experiments.

        Aligns amplitudes by timepoints and calculates averages and raw data matrix.

        Returns:
            tuple: (time_points, mean_amplitudes, all_amplitudes, files_before_treatment)
        """
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None

        # Assume all experiments have the same number of files/timepoints
        n_timepoints = self.experiments[0].get_file_count()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment() # This will be zero if no files before treatment
        time_points = self.get_timepoints_minutes()

        all_amplitudes = np.full((n_experiments, n_timepoints), np.nan, dtype=float)

        for i, experiment in enumerate(self.experiments):
            for j, spheroid_file in enumerate(experiment.files):
                metadata = spheroid_file.get_metadata()
                peak_amplitude_values = metadata['peak_amplitude_values']
                # If peak_amplitude_values is None, empty, or zero, keep as zero
                if peak_amplitude_values is None or (isinstance(peak_amplitude_values, (list, np.ndarray)) and len(peak_amplitude_values) == 0):
                    all_amplitudes[i, j] = 0.0
                elif isinstance(peak_amplitude_values, (list, np.ndarray)):
                    # Use mean for multi-peak files, scalar for single peak
                    all_amplitudes[i, j] = float(np.mean(peak_amplitude_values))
                else:
                    all_amplitudes[i, j] = float(peak_amplitude_values)
        mean_amplitudes = np.nanmean(all_amplitudes, axis=0)
        return time_points, mean_amplitudes, all_amplitudes, files_before_treatment

    def get_all_AUC(self, show_plot: bool = False):
        """
        Compute the Area Under the Curve (AUC) for post-stimulation IT traces across all experiments.

        For each experiment and its replicate files, this method:
          1. Determines an integration start point—either the end of a stimulation artifact (with linear interpolation)
             or the detected onset of the signal rise when no artifact is present.
          2. Identifies the end of the integration window by finding the first zero‐crossing after the peak (or
             the signal minimum if no crossing is found).
          3. Applies Simpson’s rule to the processed signal between these bounds.
          4. Optionally displays a diagnostic plot per file, showing raw vs. processed data, integration region,
             and key markers (stimulation window, rise start, peak, integration end).

        Args:
            show_plot (bool, optional): If True, generate a matplotlib figure for each file illustrating
                the original trace, artifact removal (or gradient detection), integration limits, and the
                filled AUC region. Defaults to True.

        Returns:
            list[list[float]] or None:
                - If experiments are present, returns a nested list where each sublist contains the AUC values
                  computed for each file in that experiment.
                - Returns None immediately if there are no experiments to process.
        """
        import numpy as np
        from scipy.integrate import simpson
        import matplotlib.pyplot as plt

        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None

        all_AUC = []
        for i, experiment in enumerate(self.experiments):
            records_AUC = []
            acq_freq = experiment.get_acquisition_frequency()
            stim_params = experiment.stim_params

            # Determine stimulation parameters
            if stim_params is not None:
                start_stim = int(stim_params['start'] * acq_freq)
                dur_stim = int(stim_params['duration'] * acq_freq)
                is_stim = True
            else:
                is_stim = False
                grad_start = 0

            for j, spheroid_file in enumerate(experiment.files):
                # Metadata and raw signal
                metadata = spheroid_file.get_metadata()
                peak_idx = self._resolve_peak_position(metadata)
                raw = spheroid_file.get_processed_data_IT()
                sig = raw.copy()

                # Optionally plot original data
                if show_plot:
                    fig, ax = plt.subplots()
                    x_all = np.arange(len(raw))
                    ax.scatter(x_all, raw, s=10, alpha=0.4, label='Original data')

                # Remove stim artifact or find gradient start
                if is_stim:
                    end_stim = start_stim + dur_stim
                    xp = [start_stim - 1, end_stim]
                    fp = [sig[xp[0]], sig[xp[1]]]
                    sig[start_stim:end_stim] = np.interp(
                        np.arange(start_stim, end_stim), xp, fp
                    )
                    start_integration = start_stim
                    if show_plot:
                        ax.plot(x_all, sig, linewidth=1, label='Processed (artifact removed)')
                else:
                    diff_IT = np.diff(sig)
                    window = diff_IT[grad_start:peak_idx]
                    sharp = np.argmax(window)
                    start_integration = grad_start + sharp + 1
                    if show_plot:
                        ax.plot(x_all, sig, linewidth=1, label='Processed (no artifact)')
                        ax.axvline(start_integration, color='green', linestyle='--', label='Integration start')

                # Find end of integration by detecting zero crossing between samples
                post_peak = sig[peak_idx:]
                # look for sign change between consecutive points
                prod = post_peak[:-1] * post_peak[1:]
                zero_cross_inds = np.where(prod <= 0)[0]
                if zero_cross_inds.size > 0:
                    # choose first crossing, adjust index to sample after peak
                    cross_idx = zero_cross_inds[0] + 1
                    end_integration = peak_idx + cross_idx
                else:
                    # fallback to minimal point
                    end_integration = peak_idx + np.argmin(post_peak)

                # Ensure valid window
                if end_integration <= start_integration:
                    end_integration = len(sig) - 1
                    if show_plot:
                        ax.plot(x_all, sig, linestyle=':', label='Fallback end (signal end)')

                # Compute AUC
                if end_integration >= start_integration:
                    x_int = np.arange(start_integration, end_integration + 1)
                    y_int = sig[start_integration:end_integration + 1]
                    auc_val = simpson(y_int)
                    if show_plot:
                        ax.fill_between(x_int, y_int, alpha=0.3, label='AUC region')
                        ax.plot(x_int, y_int, linewidth=2, label='Integration curve')
                else:
                    auc_val = 0.0

                # Finalize plot
                if show_plot:
                    ax.axvline(peak_idx, color='red', linestyle='--', label='Peak')
                    ax.axvline(end_integration, color='purple', linestyle='-.', label='Integration end')
                    ax.legend(loc='best', fontsize='small')
                    ax.set_title(f'Exp {i} File {j} — AUC={auc_val:.2f}')
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel('Signal Amplitude')
                    plt.tight_layout()
                    plt.show()

                records_AUC.append(auc_val)

            all_AUC.append(records_AUC)

        return all_AUC
    
    def legacy_get_all_AUC(self):
        """Legacy AUC computation using zero crossings.

        Calculates the AUC from the last zero before the peak to the first zero after it.

        Returns:
            list: AUC values per replicate and file.
        """
        from scipy.integrate import simpson

        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None
        
        all_AUC = []
        for i, experiment in enumerate(self.experiments):
            records_AUC = []
            for j, spheroid_file in enumerate(experiment.files):
                # Gathering position of peak
                metadata = spheroid_file.get_metadata()
                peak_amplitude_pos = metadata['peak_amplitude_positions']
                # Gathering processed data
                processed_IT = spheroid_file.get_processed_data_IT()

                # Cropping IT to find first intersect before peak
                IT_cropped_before_peak = processed_IT[:peak_amplitude_pos]
                zero_indices_before_peak = np.where(IT_cropped_before_peak == 0)[0]
                if zero_indices_before_peak.size > 0:
                    # Getting the last zero before the peak
                    mapped_intersect_before = zero_indices_before_peak[-1]
                else:
                    mapped_intersect_before = 0  # fallback to start
                # Cropping IT to find first intersect after peak
                IT_cropped_after_peak = processed_IT[peak_amplitude_pos:]
                zero_indices_after_peak = np.where(IT_cropped_after_peak == 0)[0]
                if zero_indices_after_peak.size > 0:
                    mapped_intersect = zero_indices_after_peak[0] + peak_amplitude_pos
                else:
                    min_amp_index = np.argmin(IT_cropped_after_peak)
                    mapped_intersect = min_amp_index + peak_amplitude_pos
                
                # Just in case there is a very short range
                if mapped_intersect <= 1:
                    res_AUC = 0 
                else:
                    res_AUC = simpson(processed_IT[mapped_intersect_before:mapped_intersect + 1])

                records_AUC.append(res_AUC)
            all_AUC.append(records_AUC)
        return all_AUC

    # ---------------------------------------------------------------
    # RETIRED — superseded by the joint fit (`exponential_fitting_joint`
    # / `save_exp_fit_joint`), the only exponential method still in use.
    # Path A: three sequential 1-parameter fits (k, then A, then C).
    # Kept commented out for reference; delete once the joint method has
    # been in production long enough that these are no longer needed.
    # ---------------------------------------------------------------
#     def exponential_fitting_replicated(self, replicate_time_point=0, global_peak_amplitude_position=None):
#         """Perform exponential fitting of post-peak decay curves aligned by peak.
#
#         Args:
#             replicate_time_point (int): Timepoint (file index) to use from each experiment.
#             global_peak_amplitude_position (int, optional): Optional override for alignment.
#
#         Returns:
#             tuple: (time_all, cropped_ITs, aligned_ITs, t_half, (A, k, C), (A_err, k_err, C_err), min_peak)
#         """
#         from scipy.optimize import curve_fit
#         n_experiments = len(self.experiments)
#         if n_experiments == 0:
#             return None
#         n_timepoints = self.experiments[0].get_file_time_points()
#         all_ITs = np.empty((n_experiments, n_timepoints))
#         peak_amplitude_positions = []
#
#         actual_index = replicate_time_point
#         for i, experiment in enumerate(self.experiments):
#             file = experiment.get_spheroid_file(actual_index)
#             IT_individual = file.get_processed_data_IT()
#             if IT_individual.shape[0] != n_timepoints:
#                 raise ValueError(
#                     f"Replicate {i+1} has {IT_individual.shape[0]} time points, expected {n_timepoints}.\n"
#                     "All replicates must have the same number of time points."
#                 )
#             metadata = file.get_metadata()
#             peak_amplitude_positions.append(self._resolve_peak_position(metadata))
#             all_ITs[i, :] = IT_individual
#
#         peaks = [int(p) for p in peak_amplitude_positions]
#         min_peak = np.min(peaks)
#         max_peak = np.max(peaks)
#         pre_allocated_ITs_array = np.full((n_experiments, n_timepoints - min_peak), np.nan)       
#         # Fill the pre-allocated array with the cropped ITs, starting from the peak position
#         for i, (row, peak) in enumerate(zip(all_ITs, peaks)):
#             # Now the array will have from the min peak position to the end of the time points
#             # Now this has aligned the peaks, so the first time point is the peak position for all ITs
#             cropped = row[peak:]
#             length = cropped.shape[0]
#             pre_allocated_ITs_array[i, :length] = cropped
#
#         # Crop the ITs from the end to match data sizes
#         cropped_ITs = pre_allocated_ITs_array[:, :n_timepoints-max_peak-min_peak]
#         print("Cropped ITs:",np.shape(cropped_ITs))
#         ITs_flattened = cropped_ITs.flatten()
#         print(np.shape(ITs_flattened))
#         n_cropped_timepoints = np.shape(cropped_ITs)
#
#         A = np.arange(min_peak, n_timepoints-max_peak)
#         print(np.shape(A))
#         n_post = n_timepoints - max_peak - min_peak
#         A = np.arange(n_post)               # 0,1,2,... in samples
#         time_all = np.tile(A, n_experiments)  # Repeat time point
#         print(np.shape(time_all))
#
#         mean_trace = np.mean(cropped_ITs, axis=0)
#         C0 = np.median(mean_trace[-10:])
#         A0 = np.mean(mean_trace[0:10])
#         k0 = 0.01
#         p0 = [k0]
#
#         #Fit k only, fix A0 and C0
#         def exp_decay_k_only(t, k):
#             return (A0 - C0) * np.exp(-k * t) + C0
#
#         popt_k, pcov_k = curve_fit(exp_decay_k_only, time_all, ITs_flattened, p0=[k0])
#         k_fit = popt_k[0]
#         k_err = np.sqrt(np.diag(pcov_k))[0]
#
#         tau_fit = 1 / k_fit
#         t_half = np.log(2) * tau_fit
#
#         # Fit A only, fix k and C0
#         def exp_decay_fixed_kC(t, A):
#             return (A - C0) * np.exp(-k_fit * t) + C0
#
#         popt_a, pcov_a = curve_fit(exp_decay_fixed_kC, time_all, ITs_flattened, p0=[A0])
#         A_fit = popt_a[0]
#         A_err = np.sqrt(np.diag(pcov_a))[0]
#
#         # Fit C only, fix k and A
#         def exp_decay_fixed_kA(t, C):
#             return (A_fit - C) * np.exp(-k_fit * t) + C
#
#         popt_c, pcov_c = curve_fit(exp_decay_fixed_kA, time_all, ITs_flattened, p0=[C0])
#         C_fit = popt_c[0]
#         C_err = np.sqrt(np.diag(pcov_c))[0]
#
#         # Final report
#         #print("Fit results (sequential):")
#         #print(f"k = {k_fit:.4f} ± {k_err:.4f}")
#         #print(f"A = {A_fit:.4f} ± {A_err:.4f}")
#         #print(f"C = {C_fit:.4f} ± {C_err:.4f}")
#         #print(f"Tau = {tau_fit:.4f}")
#         #print(f"t_half = {t_half:.4f}")
#
#         # Pre-allocated_ITs_array is the matrix with all data properly aligned on their peaks
#         return time_all, cropped_ITs, pre_allocated_ITs_array, t_half, (A_fit, k_fit, C_fit), (A_err, k_err, C_err), min_peak

    # ==================================================================
    #  New replicate-fitting methods (added alongside Path A / legacy).
    #  Shared model: y(t) = (A - C) exp(-k t) + C, t=0,1,... in samples,
    #  re-zeroed at each replicate's OWN peak. k is fitted per timepoint
    #  and is NEVER shared across timepoints (that is the drug effect).
    # ==================================================================

    def _gather_post_peak_traces(self, replicate_time_point):
        """Collect each replicate's post-peak IT trace at one timepoint.

        Every replicate is aligned on ITS OWN peak (``_resolve_peak_position``),
        NOT on a shared global-peak crop. No common crop is applied here: each
        returned trace keeps its full post-peak length. The joint method (which
        flattens into one vector) applies its own common crop on top of this.

        Args:
            replicate_time_point (int): file index shared across replicates.

        Returns:
            tuple: (traces, peaks, n_timepoints) where ``traces`` is a list of
            1-D float arrays (one per replicate, own post-peak length) and
            ``peaks`` is the list of int peak positions. ``None`` if empty.
        """
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None
        n_timepoints = self.experiments[0].get_file_time_points()
        traces = []
        peaks = []
        for i, experiment in enumerate(self.experiments):
            file = experiment.get_spheroid_file(replicate_time_point)
            IT_individual = file.get_processed_data_IT()
            if IT_individual.shape[0] != n_timepoints:
                raise ValueError(
                    f"Replicate {i+1} has {IT_individual.shape[0]} time points, expected {n_timepoints}.\n"
                    "All replicates must have the same number of time points."
                )
            metadata = file.get_metadata()
            peak = int(self._resolve_peak_position(metadata))
            peaks.append(peak)
            traces.append(np.asarray(IT_individual[peak:], dtype=float))
        return traces, peaks, n_timepoints

    @staticmethod
    def _tmultiplier(nu):
        """Two-sided 97.5% Student-t multiplier for ``nu`` dof (t_{nu,0.975}).

        Falls back to z=1.96 only when nu is non-finite/non-positive; for large
        nu (thousands) t.ppf already returns ~1.96 on its own.
        """
        from scipy.stats import t as t_dist
        if nu is None or not np.isfinite(nu) or nu <= 0:
            return 1.96
        return float(t_dist.ppf(0.975, nu))

    @classmethod
    def _tau_thalf_from_k(cls, k, se_k, nu):
        """Derive tau and t_half (point estimate, SE, 95% CI) from a rate k.

        All quantities are in SAMPLES (convert to seconds only at output time).
        SEs use the delta method: SE(tau) = SE(k)/k^2 and
        SE(t_half) = ln(2)*SE(k)/k^2. The 95% CI is built on k and its endpoints
        are INVERTED (the low tau endpoint comes from the high k endpoint), which
        is correct whether SE(k)/k is small (near-symmetric) or large
        (asymmetric), so the endpoint-inversion form is always used.

        A non-finite se_k (e.g. degenerate curvature in shared-k) is allowed: the
        point estimates tau/t_half are still returned, with NaN SE and NaN CI.

        Args:
            k (float): decay rate in 1/samples. Must be finite and > 0.
            se_k (float): standard error of k (>= 0), or non-finite if unknown.
            nu (float): degrees of freedom for the Student-t multiplier.

        Returns:
            dict: keys ``tau``, ``t_half``, ``se_tau``, ``se_t_half``,
            ``tau_ci`` (lo, hi), ``t_half_ci`` (lo, hi) and ``tmult`` — all in
            samples except ``tmult``.

        Raises:
            ValueError: if k is non-finite or <= 0, or if se_k is finite and < 0.
        """
        if not np.isfinite(k) or k <= 0:
            raise ValueError("k must be finite and greater than zero.")
        if np.isfinite(se_k) and se_k < 0:
            raise ValueError("se_k must be non-negative.")

        tmult = cls._tmultiplier(nu)
        tau = 1.0 / k
        t_half = LN2 / k

        if np.isfinite(se_k):
            se_tau = se_k / k**2
            se_t_half = LN2 * se_k / k**2
            k_lo = k - tmult * se_k
            k_hi = k + tmult * se_k
            # Flip: smaller k -> larger tau, so the tau interval endpoints swap.
            tau_ci_lo = (1.0 / k_hi) if k_hi > 0 else np.nan
            tau_ci_hi = (1.0 / k_lo) if k_lo > 0 else np.inf
            th_ci_lo = (LN2 / k_hi) if k_hi > 0 else np.nan
            th_ci_hi = (LN2 / k_lo) if k_lo > 0 else np.inf
        else:
            # SE unknown -> report the point estimate with NaN SE/CI.
            se_tau = se_t_half = np.nan
            tau_ci_lo = tau_ci_hi = np.nan
            th_ci_lo = th_ci_hi = np.nan

        return {
            "tau": tau, "t_half": t_half,
            "se_tau": se_tau, "se_t_half": se_t_half,
            "tau_ci": (tau_ci_lo, tau_ci_hi),
            "t_half_ci": (th_ci_lo, th_ci_hi),
            "tmult": tmult,
        }

    def exponential_fitting_joint(self, replicate_time_point=0):
        """Method 1 — adjusted Path A: joint (A, k, C) fit on pooled replicates.

        Same pooling as Path A (own-peak aligned, then flatten all replicates
        into one vector with a tiled time axis), but A, k and C are fitted
        SIMULTANEOUSLY with a single 3-parameter ``curve_fit`` on the
        (A-C)exp(-kt)+C model — not three sequential one-parameter fits. SEs
        come from the full 3x3 covariance diagonal.

        Sharing within the timepoint: one A, one k, one C for all replicates.
        Common crop length = ``n_timepoints - max_peak`` (the extra ``-min_peak``
        subtraction in Path A is a bug that discards valid tail samples).
        nu = N - 3 with N = n_reps * n_post (in the thousands, so z~1.96).
        """
        from scipy.optimize import curve_fit
        gathered = self._gather_post_peak_traces(replicate_time_point)
        if gathered is None:
            return None
        traces, peaks, n_timepoints = gathered
        n_reps = len(traces)
        max_peak = int(np.max(peaks))
        n_post = n_timepoints - max_peak  # correct common length (no extra -min_peak)
        if n_post < 4:
            raise ValueError(f"Only {n_post} post-peak samples after alignment — too few to fit.")

        cropped = np.vstack([tr[:n_post] for tr in traces])  # (n_reps, n_post)
        y = cropped.flatten()
        t = np.tile(np.arange(n_post, dtype=float), n_reps)

        mean_trace = cropped.mean(axis=0)
        A0 = float(mean_trace[:min(10, n_post)].mean())
        C0 = float(np.median(mean_trace[-min(10, n_post):]))
        k0 = 0.01

        popt, pcov = curve_fit(_model_AkC, t, y, p0=[A0, k0, C0], maxfev=20000)
        A_fit, k_fit, C_fit = (float(v) for v in popt)
        perr = np.sqrt(np.diag(pcov))
        se_k = float(perr[1])

        N = n_reps * n_post
        nu = N - 3
        result = {
            "method": "joint", "k": k_fit, "se_k": se_k,
            "A": A_fit, "C": C_fit,
            "A_se": float(perr[0]), "C_se": float(perr[2]),
            "n_used": n_reps, "N": N, "nu": nu,
            "n_post": n_post, "max_peak": max_peak, "pcov": pcov,
            # Aligned/cropped data behind the fit, so the plots can draw the
            # traces and the fit from a single call instead of re-gathering.
            "cropped_ITs": cropped,          # (n_reps, n_post), own-peak aligned
            "time_all": t,                   # tiled sample axis matching cropped.flatten()
            "min_peak": int(np.min(peaks)),
        }
        result.update(self._tau_thalf_from_k(k_fit, se_k, nu))
        return result

    # ---------------------------------------------------------------
    # RETIRED — superseded by the joint fit (`exponential_fitting_joint`
    # / `save_exp_fit_joint`), the only exponential method still in use.
    # Helpers used only by the shared-k profile-likelihood fit.
    # Kept commented out for reference; delete once the joint method has
    # been in production long enough that these are no longer needed.
    # ---------------------------------------------------------------
#     @staticmethod
#     def _second_derivative(f, x, rel=1e-2):
#         """Curvature d2f/dx2 via a 5-point least-squares parabola around ``x``.
#
#         Fitting a parabola to a small symmetric stencil is more robust to the
#         tiny numerical noise of the per-replicate lstsq than a single 3-point
#         difference.
#         """
#         h = max(abs(x) * rel, 1e-8)
#         xs = x + h * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
#         ys = np.array([f(xi) for xi in xs])
#         coeffs = np.polyfit(xs - x, ys, 2)
#         return 2.0 * coeffs[0]
#
#     @staticmethod
#     def _profile_ci_k(sse_only, k_opt, thresh, k_lo_bound, k_hi_bound):
#         """Profile-likelihood CI on k: where profiled SSE crosses ``thresh``.
#
#         Returns (k_low, k_high); either may be NaN if SSE never crosses the
#         threshold within the search bounds (interval unbounded on that side).
#         """
#         from scipy.optimize import brentq
#         g = lambda k: sse_only(k) - thresh
#         try:
#             k_low = float(brentq(g, k_lo_bound, k_opt)) if g(k_lo_bound) > 0 else np.nan
#         except (ValueError, RuntimeError):
#             k_low = np.nan
#         try:
#             k_high = float(brentq(g, k_opt, k_hi_bound)) if g(k_hi_bound) > 0 else np.nan
#         except (ValueError, RuntimeError):
#             k_high = np.nan
#         return k_low, k_high

    # ---------------------------------------------------------------
    # RETIRED — superseded by the joint fit (`exponential_fitting_joint`
    # / `save_exp_fit_joint`), the only exponential method still in use.
    # Method 2: one shared k with per-replicate A_i / C_i.
    # Kept commented out for reference; delete once the joint method has
    # been in production long enough that these are no longer needed.
    # ---------------------------------------------------------------
#     def exponential_fitting_shared_k(self, replicate_time_point=0, profile_ci=True):
#         """Method 2 — fixed-effects shared-k fit (one k, per-replicate A_i, C_i).
#
#         Within the timepoint every replicate shares a single k but keeps its own
#         A_i and C_i. Implemented by PROFILING, not a raw 2n+1-parameter optimiser:
#         for a trial k the model is LINEAR in (A_i, C_i), so with
#         X(k) = [exp(-k t), 1 - exp(-k t)] the per-replicate (A_i, C_i) are solved
#         in closed form by least squares, the SSE is summed across replicates, and
#         the 1-D profiled SSE is minimised over log k with ``minimize_scalar``.
#         This needs no A/C starting values and is faster and more robust than a
#         2n+1-param fit. Each replicate uses its OWN post-peak length (no common
#         crop — nothing is flattened into one vector).
#
#         SE(k) uses the curvature of the profiled SSE at the optimum:
#         se_k = sqrt(2*sigma2/d2), sigma2 = SSE/(N-p), p = 2n+1, N = total points.
#         (This equals pcov[0,0] of the full fit; building the 2n+1 Jacobian instead
#         would need an SVD pseudo-inverse because J^T J is near-singular here.)
#         nu = N - p. An optional asymmetric profile-likelihood CI on k is also
#         returned (preferred when the interval is wide).
#         """
#         from scipy.optimize import minimize_scalar
#         gathered = self._gather_post_peak_traces(replicate_time_point)
#         if gathered is None:
#             return None
#         traces, peaks, n_timepoints = gathered
#         n_reps = len(traces)
#         times = [np.arange(len(tr), dtype=float) for tr in traces]
#         N = int(sum(len(tr) for tr in traces))
#         p = 2 * n_reps + 1
#
#         def solve_linear(k):
#             """Closed-form (A_i, C_i) per replicate for a trial k; returns SSE too."""
#             sse = 0.0
#             A = np.empty(n_reps)
#             C = np.empty(n_reps)
#             for i, (y, t) in enumerate(zip(traces, times)):
#                 E = np.exp(-k * t)
#                 X = np.column_stack([E, 1.0 - E])   # coeffs are [A, C]
#                 beta, _res, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
#                 A[i], C[i] = beta
#                 sse += float(np.sum((y - X @ beta) ** 2))
#             return sse, A, C
#
#         def sse_only(k):
#             return solve_linear(k)[0]
#
#         k_lo_bound, k_hi_bound = 1e-5, 5.0
#         res = minimize_scalar(
#             lambda lk: sse_only(np.exp(lk)),
#             bounds=(np.log(k_lo_bound), np.log(k_hi_bound)),
#             method="bounded", options={"xatol": 1e-10},
#         )
#         k_fit = float(np.exp(res.x))
#         sse_min, A_i, C_i = solve_linear(k_fit)
#
#         dof = max(N - p, 1)
#         sigma2 = sse_min / dof
#         d2 = self._second_derivative(sse_only, k_fit)
#         se_k = float(np.sqrt(2.0 * sigma2 / d2)) if (np.isfinite(d2) and d2 > 0) else np.nan
#         nu = N - p
#
#         result = {
#             "method": "shared_k", "k": k_fit, "se_k": se_k,
#             "A": A_i, "C": C_i,               # per-replicate arrays
#             "n_used": n_reps, "N": N, "nu": nu, "p": p, "sse": sse_min,
#         }
#         result.update(self._tau_thalf_from_k(k_fit, se_k, nu))
#
#         if profile_ci:
#             from scipy.stats import chi2
#             thresh = sse_min * (1.0 + chi2.ppf(0.95, 1) / dof)
#             k_low, k_high = self._profile_ci_k(sse_only, k_fit, thresh, k_lo_bound, k_hi_bound)
#             result["prof_k_ci"] = (k_low, k_high)
#             # invert endpoints for a tau interval (flip: low tau <- high k)
#             result["tau_prof_ci"] = (
#                 (1.0 / k_high) if (np.isfinite(k_high) and k_high > 0) else np.nan,
#                 (1.0 / k_low) if (np.isfinite(k_low) and k_low > 0) else np.inf,
#             )
#         return result

    # ---------------------------------------------------------------
    # RETIRED — superseded by the joint fit (`exponential_fitting_joint`
    # / `save_exp_fit_joint`), the only exponential method still in use.
    # Method 3: per-replicate fits combined linearly over tau_i.
    # Kept commented out for reference; delete once the joint method has
    # been in production long enough that these are no longer needed.
    # ---------------------------------------------------------------
#     def exponential_fitting_two_stage(self, replicate_time_point=0,
#                                       outlier_mad=3.0, min_n_for_outlier=4, verbose=True):
#         """Method 3 — two-stage: fit each replicate, then combine tau_i (linear).
#
#         Each replicate is fitted INDEPENDENTLY with a joint 3-parameter
#         ``curve_fit`` (its own A_i, k_i, C_i) on its own post-peak trace;
#         tau_i = 1 / k_i. Two drop tiers, tracked separately:
#
#           Stage 1 — non-fit rejection (list ``rejected``):
#             no_convergence : curve_fit raised (RuntimeError / ValueError)
#             nonfinite_k    : fitted k not finite or <= 0
#             nonfinite_se   : SE(k) not finite
#           Stage 2 — MEDIAN/MAD outlier trim (list ``outliers``), ONE pass:
#             med = median(tau_i);  mad = 1.4826 * median(|tau_i - med|)
#             keep tau_i where |tau_i - med| <= outlier_mad * mad
#             Median/MAD (not mean/SD) is used deliberately: mean/SD is MASKED by
#             the very outliers being removed — several gross outliers inflate the
#             SD so they sit within k*SD of their own inflated mean. Skipped when
#             fewer than ``min_n_for_outlier`` fits survive stage 1 (too few to
#             judge) or when mad == 0.
#
#         The KEPT tau_i (n_used) are combined linearly:
#             tau_bar = mean(tau_i);  SD = std(ddof=1);  SE = SD / sqrt(n_used)
#             CI95    = tau_bar +/- t_{n_used-1, 0.975} * SE
#         """
#         from scipy.optimize import curve_fit
#         gathered = self._gather_post_peak_traces(replicate_time_point)
#         if gathered is None:
#             return None
#         traces, peaks, n_timepoints = gathered
#         n_reps = len(traces)
#
#         # ---- Stage 1: per-replicate fit; drop only genuine non-fits ----
#         tau_v, se_tau_v, k_v, A_v, C_v, idx_v = [], [], [], [], [], []
#         rejected = []   # (index, category, message)
#
#         def _reject(i, category, msg):
#             rejected.append((i, category, msg))
#             if verbose:
#                 print(f"[two-stage] tp={replicate_time_point}: rejected replicate "
#                       f"{i+1} ({category}: {msg})")
#
#         for i, tr in enumerate(traces):
#             m = len(tr)
#             t = np.arange(m, dtype=float)
#             y = np.asarray(tr, dtype=float)
#             A0 = float(y[0])
#             C0 = float(np.median(y[-min(10, m):]))
#             # Only the fit call is guarded: catch non-convergence / bad-input from
#             # curve_fit itself, NOT a bug in our own validation below (which must
#             # surface). Hence (RuntimeError, ValueError), never bare Exception.
#             try:
#                 popt, pcov = curve_fit(_model_AkC, t, y, p0=[A0, 0.01, C0], maxfev=20000)
#             except (RuntimeError, ValueError) as e:
#                 _reject(i, "no_convergence", str(e))
#                 continue
#
#             A, k, C = (float(v) for v in popt)
#             se_k = float(np.sqrt(np.diag(pcov))[1])
#             if not np.isfinite(k) or k <= 0:
#                 _reject(i, "nonfinite_k", f"k={k}")
#                 continue
#             if not np.isfinite(se_k):
#                 _reject(i, "nonfinite_se", f"se_k={se_k}")
#                 continue
#
#             tau_v.append(1.0 / k)
#             se_tau_v.append(se_k / k**2)   # delta method, per replicate
#             k_v.append(k)
#             A_v.append(A)
#             C_v.append(C)
#             idx_v.append(i)
#
#         tau_v = np.asarray(tau_v)
#         se_tau_v = np.asarray(se_tau_v)
#         n_valid = len(tau_v)
#
#         # ---- Stage 2: one-pass MEDIAN/MAD outlier trim on the surviving tau_i ----
#         # MAD, not mean/SD: mean/SD is masked by the very outliers we are removing.
#         outliers = []                      # (index, tau_samples)
#         keep = np.ones(n_valid, dtype=bool)
#         if n_valid >= min_n_for_outlier:
#             med = float(np.median(tau_v))
#             mad = 1.4826 * float(np.median(np.abs(tau_v - med)))
#             if np.isfinite(mad) and mad > 0:   # mad==0 -> no scale to judge; keep all
#                 keep = np.abs(tau_v - med) <= outlier_mad * mad
#             for j in np.where(~keep)[0]:
#                 gi = idx_v[j]
#                 outliers.append((gi, float(tau_v[j])))
#                 if verbose:
#                     print(f"[two-stage] tp={replicate_time_point}: outlier replicate "
#                           f"{gi+1} (tau={tau_v[j]:.1f} samples, > {outlier_mad} MAD from median)")
#
#         used_idx = [idx_v[j] for j in range(n_valid) if keep[j]]
#         tau_i = tau_v[keep]
#         se_tau_i = se_tau_v[keep]
#         k_i = np.asarray([k_v[j] for j in range(n_valid) if keep[j]])
#         A_i = np.asarray([A_v[j] for j in range(n_valid) if keep[j]])
#         C_i = np.asarray([C_v[j] for j in range(n_valid) if keep[j]])
#         n_used = len(tau_i)
#         n_outliers = len(outliers)
#
#         # ---- Combine linearly on the kept tau_i ----
#         if n_used >= 2:
#             tau_bar = float(np.mean(tau_i))
#             s = float(np.std(tau_i, ddof=1))
#             se_tau = s / np.sqrt(n_used)
#             nu = n_used - 1
#             tmult = self._tmultiplier(nu)
#             tau_ci = (tau_bar - tmult * se_tau, tau_bar + tmult * se_tau)
#             status = "ok"
#         elif n_used == 1:
#             tau_bar = float(tau_i[0])
#             s = se_tau = np.nan
#             nu = 0
#             tmult = np.nan
#             tau_ci = (np.nan, np.nan)
#             status = "low_n: n=1 (no interval)"
#         else:
#             tau_bar = s = se_tau = np.nan
#             nu = 0
#             tmult = np.nan
#             tau_ci = (np.nan, np.nan)
#             status = "no_valid_fits"
#
#         t_half_bar = LN2 * tau_bar
#         se_t_half = LN2 * se_tau
#         t_half_ci = (LN2 * tau_ci[0], LN2 * tau_ci[1])
#
#         return {
#             "method": "two_stage", "status": status,
#             "k": (1.0 / tau_bar) if (np.isfinite(tau_bar) and tau_bar != 0) else np.nan,
#             "tau": tau_bar, "se_tau": se_tau, "std_tau": s,   # std_tau = real std(kept tau_i)
#             "t_half": t_half_bar, "se_t_half": se_t_half,
#             "tau_ci": tau_ci, "t_half_ci": t_half_ci, "tmult": tmult,
#             "tau_i": tau_i, "se_tau_i": se_tau_i, "k_i": k_i,
#             "A": A_i, "C": C_i,
#             "used_idx": used_idx,
#             "rejected": rejected,                             # stage-1 non-fits
#             "outliers": outliers,                             # stage-2 SD outliers (idx, tau)
#             # back-compat alias: everything not used, with a reason
#             "dropped": [(i, f"{cat}: {msg}") for i, cat, msg in rejected]
#                        + [(i, f"outlier: tau={tau:.1f} samples") for i, tau in outliers],
#             "n_used": n_used, "n_valid": n_valid, "n_reps": n_reps,
#             "n_rejected": len(rejected), "n_outliers": n_outliers, "nu": nu,
#         }

    # ---------------------------------------------------------------
    # RETIRED — superseded by the joint fit (`exponential_fitting_joint`
    # / `save_exp_fit_joint`), the only exponential method still in use.
    # Between/within-replicate variance split; fed the two-stage CSV only.
    # Kept commented out for reference; delete once the joint method has
    # been in production long enough that these are no longer needed.
    # ---------------------------------------------------------------
#     def replicate_variance_diagnostic(self, replicate_time_point=0):
#         """Decompose tau spread into between- vs within-replicate parts.
#
#         Built from the KEPT two-stage per-replicate results at this timepoint
#         (consistent with the linear mean estimator), all in SAMPLES:
#             s           = std(tau_i, ddof=1)             # between + within
#             se_bar      = sqrt(mean(SE(tau_i)^2))        # within only
#             sigma_b_hat = sqrt(max(0, s^2 - se_bar^2))   # between only (clamped)
#         Rule of thumb for the caller: s ~ se_bar => the shared-k residual SE is
#         trustworthy; s >> se_bar => report the two-stage (between-replicate)
#         interval instead.
#
#         Caveat: SavGol/Butterworth filtering makes each fit's residuals
#         autocorrelated, which shrinks the individual SE(tau_i) and hence se_bar,
#         biasing this test toward "spread wins". When it is ambiguous, prefer the
#         two-stage interval.
#         """
#         ts = self.exponential_fitting_two_stage(replicate_time_point, verbose=False)
#         if ts is None:
#             return None
#         tau_i = np.asarray(ts["tau_i"], dtype=float)
#         se_tau_i = np.asarray(ts["se_tau_i"], dtype=float)
#         n_used = ts["n_used"]
#         if n_used < 2:
#             return {"s": np.nan, "se_bar": np.nan, "sigma_b_hat": np.nan, "n_used": n_used}
#         s = float(np.std(tau_i, ddof=1))
#         se_bar = float(np.sqrt(np.mean(se_tau_i ** 2)))
#         sigma_b_hat = float(np.sqrt(max(0.0, s**2 - se_bar**2)))
#         return {"s": s, "se_bar": se_bar, "sigma_b_hat": sigma_b_hat, "n_used": n_used}

    def get_tau_over_time(self):
        """Extract tau (decay time constant) at each replicate time point.

        Uses the JOINT fit (``exponential_fitting_joint``) — the only exponential
        method still active. tau and its SE come straight from that fit's delta
        -method values, both in SAMPLES (callers convert to seconds if needed).

        Returns:
            tuple: (tau_list, tau_error_list)
        """
        n_files = self.experiments[0].get_file_count()
        tau_list = []
        tau_err_list = []
        for t in range(n_files):
            try:
                result = self.exponential_fitting_joint(replicate_time_point=t)
                if result is None:
                    raise ValueError("joint fit returned no result")
                tau_list.append(result["tau"])
                tau_err_list.append(result["se_tau"])
            except Exception as e:
                print(f"[tau over time] timepoint {t}: skipping ({e})")
                tau_list.append(np.nan)
                tau_err_list.append(np.nan)
        return tau_list, tau_err_list
    
    # ---------------------------------------------------------------
    # RETIRED — superseded by the joint fit (`exponential_fitting_joint`
    # / `save_exp_fit_joint`), the only exponential method still in use.
    # Fed save_all_exponential_fitting_params, which is retired too.
    # Kept commented out for reference; delete once the joint method has
    # been in production long enough that these are no longer needed.
    # ---------------------------------------------------------------
#     def get_exponential_fit_params_over_time(self):
#         """
#         Runs exponential_fitting_replicated for each replicate time point,
#         collects A, tau, C, t_half and their errors, and returns them as a 2D numpy array.
#         Columns: A_fit, A_error, tau_fit, tau_error, C_fit, C_error, t_half, t_half_error
#         Rows: replicate time points
#         The errors here are the standard error of the different quantities, in other words, 
#         what is computed is the one-sigma (≈ 68 % coverage) standard error.
#
#         Returns:
#             np.ndarray: A matrix of shape (n_timepoints, 16) with values and uncertainties.
#         """
#         n_files = self.experiments[0].get_file_count()
#         n_reps = len(self.experiments)
#         z95 = 1.96
#
#         results = []
#         for t in range(n_files):
#             try:
#                 _, _, _, t_half, fit_vals, fit_errs, _ = self.exponential_fitting_replicated(replicate_time_point=t)
#                 A_fit, k_fit, C_fit = fit_vals
#                 A_SE,  k_SE,   C_SE = fit_errs
#
#                 tau_fit   = 1.0 / k_fit
#                 tau_SE    = abs(k_SE   / k_fit**2)
#                 t_half    = np.log(2) * tau_fit
#                 t_half_SE = np.log(2) * tau_SE
#
#                 # convert SE to sample SD
#                 A_SD      = A_SE      * np.sqrt(n_reps)
#                 tau_SD    = tau_SE    * np.sqrt(n_reps)
#                 C_SD      = C_SE      * np.sqrt(n_reps)
#                 t_half_SD = t_half_SE * np.sqrt(n_reps)
#
#                 # 95% CI half‐widths
#                 A_CI95      = z95 * A_SE
#                 tau_CI95    = z95 * tau_SE
#                 C_CI95      = z95 * C_SE
#                 t_half_CI95 = z95 * t_half_SE
#
#                 results.append([
#                     A_fit,   A_SE,   A_SD,   A_CI95,
#                     tau_fit, tau_SE, tau_SD, tau_CI95,
#                     C_fit,   C_SE,   C_SD,   C_CI95,
#                     t_half,  t_half_SE, t_half_SD, t_half_CI95
#                 ])
#
#             except Exception as e:
#                 results.append([np.nan]*16)
#         return np.array(results)
    
    # ---------------------------------------------------------------
    # RETIRED — superseded by the joint fit (`exponential_fitting_joint`
    # / `save_exp_fit_joint`), the only exponential method still in use.
    # Legacy global-peak crop (no per-replicate peak alignment).
    # Kept commented out for reference; delete once the joint method has
    # been in production long enough that these are no longer needed.
    # ---------------------------------------------------------------
#     def exponential_fitting_replicated_legacy(self, replicate_time_point = 0, global_peak_amplitude_position=None):
#         """
#         Legacy Function - No Longer Supporter. This function performs exponential fitting of IT curves without aligning the peaks across replicates.
#
#         Args:
#             replicate_time_point (int): Index of the replicate time point (e.g., 3 for first post-treatment).
#             global_peak_amplitude_position (int, optional): Optional override for peak alignment index.
#
#         Returns:
#             tuple: A tuple containing:
#                 - time_all (np.ndarray): Time vector repeated across replicates.
#                 - ITs_flattened (np.ndarray): Flattened IT signal array.
#                 - t_half (float): Estimated half-life of the decay.
#                 - popt (list): Fitted parameters [A, tau, C].
#                 - pcov (np.ndarray): Covariance matrix of the fit.
#                 - A_fit (float): Fitted amplitude.
#                 - tau_fit (float): Fitted decay constant.
#                 - C_fit (float): Fitted baseline value.
#         """
#
#         from scipy.optimize import curve_fit
#
#         n_experiments = len(self.experiments)
#         if n_experiments == 0:
#             return None, None, None, None
#
#         # Assume all experiments have the same number of files/timepoints
#         n_timepoints = self.experiments[0].get_file_time_points()
#         files_before_treatment = self.experiments[0].get_number_of_files_before_treatment() # This will be zero if no files before treatment
#
#         all_ITs = np.empty((n_experiments, n_timepoints))
#         peak_amplitude_positions = []
#
#         actual_index = replicate_time_point + files_before_treatment
#         for i, experiment in enumerate(self.experiments):
#             file = experiment.get_spheroid_file(actual_index)
#             IT_individual = file.get_processed_data_IT()
#             metadata = file.get_metadata()
#             peak_amplitude_positions.append(metadata["peak_amplitude_positions"])
#             all_ITs[i, :] = IT_individual
#
#         print(peak_amplitude_positions)
#         latest_peak_amplitude_positions = np.max(peak_amplitude_positions)
#         if global_peak_amplitude_position is None:
#             global_peak_amplitude_position = latest_peak_amplitude_positions
#         else:
#             global_peak_amplitude_position = int(global_peak_amplitude_position)
#         print(global_peak_amplitude_position)
#         cropped_ITs = all_ITs[:, global_peak_amplitude_position:]
#
#         ITs_flattened = cropped_ITs.flatten()
#         n_cropped_timepoints = np.shape(cropped_ITs)
#
#         A = np.arange(global_peak_amplitude_position, n_timepoints)
#         time_all = np.tile(A, n_experiments)  # Repeat time point
#
#         #print("Len ITs Flattened:", len(ITs_flattened))
#         #print("ITs_Flattened", np.shape(ITs_flattened))
#         #print("Time All", np.shape(time_all))
#
#         # Improved initial guess for parameters
#         # A: amplitude (difference between max and min of cropped ITs)
#         # tau: decay constant (guess as 1/3 of the time range)
#         # C: baseline (last value of the mean trace)
#         mean_trace = np.mean(cropped_ITs, axis=0)
#         A0 = float(np.max(mean_trace) - np.min(mean_trace))
#         tau0 = (n_timepoints - global_peak_amplitude_position) / 3.0
#         C0 = float(mean_trace[-1])
#         p0 = [A0, tau0, C0]
#
#         print(f"Initial guess: A={A0:.2f}, tau={tau0:.2f}, C={C0:.2f}")
#
#         # Fit
#         popt, pcov = curve_fit(exp_decay, time_all, ITs_flattened, p0=p0)
#
#         # Extract parameter estimates and standard errors
#         A_fit, tau_fit, C_fit = popt
#         perr = np.sqrt(np.diag(pcov))  # Approximate symmetric 1-sigma CI
#
#         print(f"Fit results:")
#         print(f"A   = {A_fit:.2f} ± {perr[0]:.2f}")
#         print(f"tau = {tau_fit:.2f} ± {perr[1]:.2f}")
#         print(f"C   = {C_fit:.2f} ± {perr[2]:.2f}")
#
#         t_half = np.log(2) * tau_fit
#
#         return time_all, ITs_flattened, t_half, popt, pcov,  A_fit, tau_fit, C_fit

    def plot_exponential_fit_aligned(self, replicate_time_point=0, save_path=None):
        """
        Plot the exponential decay fit on peak-aligned IT curves with a 95% confidence interval.

        Drawn by the shared ``draw_exponential_fit`` so the exported figure is
        identical to the one shown on the Results page. Previously this drew a
        points x-axis with a mean +/- SD band while the on-screen version drew a
        seconds x-axis with the raw data scattered, so the two did not match.

        Args:
            replicate_time_point (int): Index of the replicate time point to analyze.
            save_path (str, optional): File path to save the figure. If None, shows the plot.

        Returns:
            tuple: A tuple containing the matplotlib figure and axis objects.
        """
        import matplotlib.pyplot as plt

        result = self.exponential_fitting_joint(replicate_time_point)
        if result is None:
            return None, None

        freq = self.experiments[0].get_acquisition_frequency()
        fig, ax = plt.subplots(figsize=(10, 6))
        draw_exponential_fit(ax, result, freq)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            fig.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        return fig, ax


    def plot_tau_over_time(self, save_path=None):
        """
        Plot tau (the exponential decay constant) over replicate time points.

        Args:
            save_path (str, optional): Path to save the figure. If None, shows the plot.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        tau_list, tau_err_list = self.get_tau_over_time()
        time_points = self.get_timepoints_minutes()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment() or 0

        plt.figure(figsize=(10, 6))
        plt.errorbar(time_points, tau_list, yerr=tau_err_list, fmt='o-', capsize=4, color='C1', label='Tau (decay constant)')
        if files_before_treatment > 0:
            plt.axvline(x=0, color='red', linestyle='--', label='Treatment Start')
        plt.xlabel("Time (minutes)")
        plt.ylabel("Tau (decay constant)")
        plt.title("Exponential Decay Tau Over Time Points")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            plt.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # ---------------------------------------------------------------
    # RETIRED — superseded by the joint fit (`exponential_fitting_joint`
    # / `save_exp_fit_joint`), the only exponential method still in use.
    # Plot for the legacy fit above.
    # Kept commented out for reference; delete once the joint method has
    # been in production long enough that these are no longer needed.
    # ---------------------------------------------------------------
#     def plot_exponential_fit_with_CI_legacy(self, replicate_time_point=0, global_peak_position=None):
#         """
#         Legacy Function - No Longer in Use, This functions plots an exponential fit and confidence interval over raw IT traces.
#
#         Args:
#             replicate_time_point (int): Index of the replicate time point.
#             global_peak_position (int, optional): Override for alignment if specified.
#
#         Returns:
#             None
#         """
#         import matplotlib.pyplot as plt
#         from scipy.stats import t
#         import numpy as np
#
#         # Run the fitting function to get values and data
#         time_all, ITs_flattened, t_half, popt, pcov, A_fit, tau_fit, C_fit = self.exponential_fitting_replicated_legacy(replicate_time_point=replicate_time_point)
#         A_fit, tau_fit, C_fit = popt
#         perr = np.sqrt(np.diag(pcov))  # 1-sigma CI
#
#         # Retrieve full ITs and metadata again for plotting individual traces
#         n_experiments = len(self.experiments)
#         file_duration = self.experiments[0].get_file_length()
#         n_timepoints = self.experiments[0].get_file_time_points()
#         files_before_treatment = self.experiments[0].get_number_of_files_before_treatment()
#         actual_index = replicate_time_point + files_before_treatment
#
#         all_ITs = np.empty((n_experiments, n_timepoints))
#         peak_positions = []
#
#         for i, experiment in enumerate(self.experiments):
#             file = experiment.get_spheroid_file(actual_index)
#             IT_individual = file.get_processed_data_IT()
#             metadata = file.get_metadata()
#             peak_pos = metadata.get("peak_amplitude_positions")
#             peak_positions.append(peak_pos)
#             all_ITs[i, :] = IT_individual
#
#         global_peak_position = int(np.max(peak_positions))
#         if global_peak_position is None:
#             global_peak_position = int(np.max(peak_positions))
#         else:
#             global_peak_position = global_peak_position
#         # Time array for full profile
#         full_time = np.arange(n_timepoints)
#
#         # Time for fitting and prediction
#         t_fit = np.linspace(global_peak_position, n_timepoints - 1, 500)
#         y_fit = A_fit * np.exp(-t_fit / tau_fit) + C_fit
#
#         # 95% CI using Jacobian
#         dof = max(0, len(time_all) - len(popt))
#         t_val = t.ppf(0.975, dof)
#
#         J = np.empty((len(t_fit), 3))
#         J[:, 0] = np.exp(-t_fit / tau_fit)
#         J[:, 1] = A_fit * t_fit / tau_fit**2 * np.exp(-t_fit / tau_fit)
#         J[:, 2] = 1
#
#         ci = np.sqrt(np.sum((J @ pcov) * J, axis=1)) * t_val
#         y_lower = y_fit - ci
#         y_upper = y_fit + ci
#
#         # Plot
#         plt.figure(figsize=(10, 6))
#
#         # Plot each replicate I-T trace
#         for i in range(n_experiments):
#             plt.plot(full_time, all_ITs[i, :], alpha=0.4, linewidth=1, label=f"Replicate {i+1}" if i == 0 else None)
#
#         # Plot fit and CI
#         plt.plot(t_fit, y_fit, label='Exponential Fit', color='red', linewidth=2)
#         plt.fill_between(t_fit, y_lower, y_upper, color='red', alpha=0.3, label='95% CI')
#
#         # Mark peak and t_half
#         plt.axvline(global_peak_position, color='orange', linestyle='--', label=f"Peak @ {global_peak_position}")
#         plt.axvline(global_peak_position + int(t_half), color='purple', linestyle=':', label=f"t_half ≈ {t_half:.2f}")
#
#         # Plot peak positions of each replicate
#         for i, peak_pos in enumerate(peak_positions):
#             if isinstance(peak_pos, (np.ndarray, list)) and len(peak_pos) > 0:
#                 pos = int(np.max(peak_pos))
#             else:
#                 pos = int(peak_pos)
#             plt.scatter(pos, all_ITs[i, pos], color='red', marker='x', s=10, label='Peak Position' if i == 0 else None, zorder=5)
#
#         plt.xlabel("Time Points (seconds)")
#         # Set ticks at every 10 seconds => every 100 time points
#         tick_locs = np.arange(0, 601, 100)       # 0, 100, 200, ..., 600
#         tick_labels = [str(int(x / 10)) for x in tick_locs]  # convert to seconds: 0, 10, ..., 60
#         plt.xticks(tick_locs, tick_labels, fontsize=10)
#         plt.ylabel("Current (nA)")
#         plt.title("Replicate I-T Profiles with Exponential Fit & CI")
#         plt.legend()
#         plt.grid(False)
#         plt.tight_layout()
#         plt.show()

    def plot_amplitudes_over_time_single_experiment(self, experiment_index=0, save_path=None):
        """
        Plot amplitudes over time for a single experiment and mark treatment start.

        Args:
            experiment_index (int): Index of the experiment to plot.
            save_path (str, optional): File path to save the figure. If None, shows the plot.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        time_points, amplitudes, files_before_treatment = self.amplitudes_over_time_single_experiment(experiment_index=experiment_index)

        # Baseline files carry negative times, so the treatment starts at zero.
        treatment_time = 0.0

        # Split data
        before_treatment_x = time_points[:files_before_treatment]
        before_treatment_y = amplitudes[:files_before_treatment]

        after_treatment_x = time_points[files_before_treatment:]
        after_treatment_y = amplitudes[files_before_treatment:]

        plt.figure(figsize=(10, 6))

        # Plot before and after separately
        plt.plot(before_treatment_x, before_treatment_y, label='Pre-Treatment', color='blue')
        plt.plot(after_treatment_x, after_treatment_y, label='Post-Treatment', color='green')

        # Vertical line for treatment start
        if files_before_treatment > 0:
            plt.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')

        # Optional scatter for emphasis
        plt.scatter(time_points, amplitudes, color='black', s=20, alpha=0.6)

        # Scientific styling
        plt.xlabel('Time (minutes)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.xticks(self._time_ticks(time_points), fontsize=10)
        plt.title('Amplitude Over Time Relative to Treatment', fontsize=14)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            plt.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    
    def plot_mean_amplitudes_over_time(self, save_path=None):
        """
        Plot the mean amplitude over time across all experiments with standard deviation.

        Args:
            save_path (str, optional): Path to save the figure. If None, shows the plot.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        time_points, mean_amplitudes, all_amplitudes, files_before_treatment = self.amplitudes_over_time_all_experiments()
        all_amplitudes = np.array(all_amplitudes, dtype=float)
        std_amplitudes = np.nanstd(all_amplitudes, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(time_points, mean_amplitudes, label='Mean Amplitude', color='purple')
        plt.fill_between(time_points, mean_amplitudes - std_amplitudes, mean_amplitudes + std_amplitudes,
                         color='purple', alpha=0.2, label='SD')
        # Baseline files carry negative times, so the treatment starts at zero.
        if files_before_treatment > 0:
            plt.axvline(x=0, color='red', linestyle='--', label='Treatment Start')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Amplitude')
        plt.title('Mean Amplitude Over Time (All Experiments)')
        plt.legend()
        plt.xticks(self._time_ticks(time_points), fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            plt.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def plot_all_amplitudes_over_time(self, save_path=None):
        """
        Plot amplitude traces for all experiments over time.

        Args:
            save_path (str, optional): Path to save the figure. If None, shows the plot.

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        
        time_points, mean_amplitudes, all_amplitudes, files_before_treatment = self.amplitudes_over_time_all_experiments()
        all_amplitudes = np.array(all_amplitudes, dtype=float)

        plt.figure(figsize=(10, 6))
        for i, amplitudes in enumerate(all_amplitudes):
            plt.plot(time_points, amplitudes, label=f'Experiment {i+1}', alpha=0.7)
        # Baseline files carry negative times, so the treatment starts at zero.
        if files_before_treatment > 0:
            plt.axvline(x=0, color='red', linestyle='--', label='Treatment Start')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Amplitude')
        plt.title('Amplitudes Over Time (All Experiments)')
        plt.legend()
        plt.xticks(self._time_ticks(time_points), fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            plt.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def plot_first_stim_amplitudes(self, save_path=None):
        """
        Plot the unnormalized amplitude from the first stimulation of each replicate.

        Args:
            save_path (str, optional): Path to save the figure. If None, shows the plot.

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        import numpy as np

        amplitudes = self.amplitudes_first_stim()  # List of lists (one per replicate)
        if amplitudes is None or len(amplitudes) == 0:
            print("No amplitudes to plot.")
            return

        # Flatten in case each amplitude is a list (e.g., [ [1.2], [1.1], ... ])
        flat_amps = [a[0] if isinstance(a, (list, np.ndarray)) and len(a) > 0 else np.nan for a in amplitudes]
        n_replicates = len(flat_amps)
        x = np.arange(1, n_replicates + 1)

        plt.figure(figsize=(8, 6))
        plt.bar(x, flat_amps, color='skyblue', edgecolor='k', alpha=0.8, label='First Stimulation Amplitude')
        plt.scatter(x, flat_amps, color='blue', zorder=5)

        # Mean and std
        mean_amp = np.nanmean(flat_amps)
        std_amp = np.nanstd(flat_amps)
        plt.axhline(mean_amp, color='red', linestyle='--', label=f'Mean = {mean_amp:.2f}')
        plt.fill_between([0, n_replicates+1], mean_amp-std_amp, mean_amp+std_amp, color='red', alpha=0.15, label='±1 SD')

        plt.xlabel("Replicate", fontsize=14)
        plt.ylabel("Amplitude (nA)", fontsize=14)
        plt.title("First Stimulation Amplitudes Across Replicates", fontsize=16)
        plt.xticks(x, [f"Rep {i}" for i in x])
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            plt.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def plot_mean_ITs(self, save_path=None):
        """
        Plots the mean IT profiles over replicates, highlighting files before treatment and marking the first file after treatment.
        
        Args:
            mean_ITs (np.ndarray): Matrix of mean IT profiles (rows = files, columns = time points).
            files_before_treatment (int): Number of files before treatment.
            title_suffix (str): Optional suffix for the plot title.
        """
        import matplotlib.pyplot as plt

        mean_ITs = self.average_IT_over_replicates()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment()  # Assuming all experiments have the same number of files before treatment

        # Number of files (rows in mean_ITs)
        n_files = mean_ITs.shape[0]

        # Time points (columns in mean_ITs), converted from samples to seconds
        freq = self.experiments[0].get_acquisition_frequency()
        time_points = np.arange(mean_ITs.shape[1]) / freq

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot files before treatment
        for i in range(files_before_treatment):
            plt.plot(time_points, mean_ITs[i, :], label=f"File {i+1} (Before Treatment)", color="blue", alpha=0.7)

        # Plot files after treatment
        for i in range(files_before_treatment, n_files):
            plt.plot(time_points, mean_ITs[i, :], label=f"File {i+1} (After Treatment)", color="green", alpha=0.7)

        # Highlight the first file after treatment
        if files_before_treatment < n_files:
            plt.plot(time_points, mean_ITs[files_before_treatment, :], label="First File After Treatment", color="red", linewidth=2)

        # Add labels, title, and legend
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Mean IT (nA)", fontsize=14)
        plt.title(f"Mean IT Profiles Over Replicates", fontsize=16)
        plt.legend(fontsize=10, loc="upper right")
        plt.grid(False)

        # Show the plot
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            plt.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_unprocessed_first_ITs(self, save_path=None):
        """
        Plot the unprocessed first IT trace from each replicate.

        Args:
            save_path (str, optional): Path to save the figure. If None, shows the plot.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        # Get the unprocessed first ITs
        first_ITs = self.non_normalized_first_ITs()

        # Number of replicates (rows in first_ITs)
        n_replicates = first_ITs.shape[0]

        # Time points (y-axis: seconds)
        time_points = np.linspace(0, self.experiments[0].get_file_length(), first_ITs.shape[1])

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot each replicate
        for i in range(n_replicates):
            plt.plot(time_points,first_ITs[i, :], label=f"Replicate {i+1}", alpha=0.7)

        # Add labels, title, and legend
        plt.ylabel("Amplitude (nA)", fontsize=14)
        plt.xlabel("Time (seconds)", fontsize=14)
        plt.title("Unprocessed First ITs of Replicates", fontsize=16)
        plt.legend(fontsize=10, loc="upper right")
        plt.grid(False)

        # Show the plot
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            plt.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_AUC(self, experiment_index=0, file_index=0, save_path=None):
        """
        Plot the IT trace for a specific experiment and file and highlight the AUC region.

        Args:
            experiment_index (int): Index of the experiment to plot.
            file_index (int): Index of the file within the experiment.
            save_path (str, optional): File path to save the figure. If None, shows the plot.

        Returns:
            tuple: A tuple with (fig, ax, auc_value) for further use.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.integrate import simpson

        # Extract data for the specified experiment and file
        experiment = self.experiments[experiment_index]
        spheroid_file = experiment.get_spheroid_file(file_index)
        processed_IT = spheroid_file.get_processed_data_IT()
        metadata = spheroid_file.get_metadata()
        peak_pos = metadata['peak_amplitude_positions']

        # Identify the first zero-crossing after the peak
        cropped_IT = processed_IT[peak_pos:]
        zero_indices = np.where(cropped_IT == 0)[0]
        if zero_indices.size > 0:
            first_zero = zero_indices[0] + peak_pos
        else:
            first_zero = len(processed_IT)

        # Calculate the area under the curve up to the first zero-crossing
        auc_value = simpson(processed_IT[:first_zero])

        # Prepare the plot
        x = np.arange(len(processed_IT))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, processed_IT, label='Processed IT')
        ax.fill_between(x[:first_zero], processed_IT[:first_zero], color='C1', alpha=0.3, label='AUC region')
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Current (nA)')
        ax.set_title('AUC Calculation')

        # Annotate the AUC value on the plot
        text_x = 0.05 * len(processed_IT)
        text_y = np.max(processed_IT) * 0.9
        ax.text(
            text_x,
            text_y,
            f'AUC = {auc_value:.2f}',
            fontsize=12,
            color='black',
            bbox=dict(facecolor='white', alpha=0.6)
        )
        ax.legend(frameon=False)
        fig.tight_layout()

        # Save or show the figure
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            fig.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        # 7) Return the figure, axis, and calculated AUC in case further processing is desired
        return fig, ax, auc_value

if __name__ == "__main__":
    import time
    start_time = time.time()
    # Example usage
    folder_first_experiment = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241111_batch1_n1_Sert"
    #folder_first_experiment = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241111_batch1_n1_Sert"
    filepaths_first_experiment = [os.path.join(folder_first_experiment, f) for f in os.listdir(folder_first_experiment) if f.endswith('.txt')]
    experiment_one = SpheroidExperiment(filepaths_first_experiment, treatment="Sertraline")
    experiment_one.run()

    folder_second_experiment = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241115_batch1_n2_Sert"
    #folder_second_experiment = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241115_batch1_n2_Sert"
    filepaths_second_experiment = [os.path.join(folder_second_experiment, f) for f in os.listdir(folder_second_experiment) if f.endswith('.txt')]   
    experiment_two = SpheroidExperiment(filepaths_second_experiment, treatment="Sertraline")
    experiment_two.run()

    folder_third_experiment = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241116_batch1_n3_Sert"
    #folder_third_experiment = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241116_batch1_n3_Sert"
    filepaths_third_experiment = [os.path.join(folder_third_experiment, f) for f in os.listdir(folder_third_experiment) if f.endswith('.txt')]
    experiment_three = SpheroidExperiment(filepaths_third_experiment, treatment="Sertraline")
    experiment_three.run()

    folder_fourth_experiment = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241128_batch2_n4_Sert"
    #folder_fourth_experiment = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241128_batch2_n4_Sert"
    filepaths_fourth_experiment = [os.path.join(folder_fourth_experiment, f) for f in os.listdir(folder_fourth_experiment) if f.endswith('.txt')]
    experiment_four = SpheroidExperiment(filepaths_fourth_experiment, treatment="Sertraline")
    experiment_four.run()

    group_analysis = GroupAnalysis()
    group_analysis.add_experiment(experiment_one, experiment_two, experiment_three, experiment_four)

    time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    #group_analysis.plot_mean_ITs()
    group_analysis.plot_AUC()
    #group_analysis.exponential_fitting_replicated()
    #group_analysis.plot_exponential_fit_aligned(replicate_time_point=0)
    #group_analysis.plot_unprocessed_first_ITs()
    #group_analysis.plot_mean_amplitudes_over_time()
    #group_analysis.plot_all_amplitudes_over_time()
    #group_analysis.amplitudes_first_stim()
    #group_analysis.plot_first_stim_amplitudes()
    #group_analysis.plot_tau_over_time()
    #group_analysis.get_all_reuptake_curves()
    #params_matrix = group_analysis.get_exponential_fit_params_over_time()
