from core.spheroid_experiment import SpheroidExperiment
from core.processing.exponentialdecay import exp_decay
from core.processing.normalize import Normalize
import os
import numpy as np

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

    def apply_calibration_to_all_experiments(self, slope, intercept):
        for experiment in self.experiments:
            experiment.apply_calibration(slope, intercept)

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
                peak_amplitude_positions.append((metadata["peak_amplitude_positions"]))
                all_ITs[i*file_count+j, :] = IT_individual

        # Turning peak_amplitude_positions into a list of integers
        # Turning peak_amplitude_positions into a list of integers
        try:
            peaks = [p.item() for p in peak_amplitude_positions]
        except AttributeError:
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
        time_points = np.linspace(
            0,
            self.experiments[0].get_time_between_files() * (n_timepoints - 1),
            n_timepoints
        )
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
        experiment = group_analysis.get_single_experiments(experiment_index)
        files_before_treatment = experiment.get_number_of_files_before_treatment()
        amplitudes = []
        time_points = np.linspace(0, experiment.get_time_between_files() * (experiment.get_file_count() - 1), experiment.get_file_count())
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
        time_points = np.linspace(
            0,
            self.experiments[0].get_time_between_files() * (n_timepoints - 1),
            n_timepoints
        )

        all_amplitudes = np.full((n_experiments, n_timepoints), np.nan, dtype=float)

        for i, experiment in enumerate(self.experiments):
            for j, spheroid_file in enumerate(experiment.files):
                metadata = spheroid_file.get_metadata()
                peak_amplitude_values = metadata['peak_amplitude_values']
                # If peak_amplitude_values is None, empty, or zero, keep as zero
                if peak_amplitude_values is None or (isinstance(peak_amplitude_values, np.ndarray) and peak_amplitude_values.size == 0):
                    all_amplitudes[i, j] = 0.0
                elif isinstance(peak_amplitude_values, np.ndarray):
                    val = float(peak_amplitude_values[0]) if peak_amplitude_values.size == 1 else float(peak_amplitude_values)
                    all_amplitudes[i, j] = val if val != 0 else 0.0
                else:
                    val = float(peak_amplitude_values)
                    all_amplitudes[i, j] = val if val != 0 else 0.0
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
                peak_idx = metadata['peak_amplitude_positions']
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

    def exponential_fitting_replicated(self, replicate_time_point=0, global_peak_amplitude_position=None):
        """Perform exponential fitting of post-peak decay curves aligned by peak.

        Args:
            replicate_time_point (int): Timepoint (file index) to use from each experiment.
            global_peak_amplitude_position (int, optional): Optional override for alignment.

        Returns:
            tuple: (time_all, cropped_ITs, aligned_ITs, t_half, (A, k, C), (A_err, k_err, C_err), min_peak)
        """
        from scipy.optimize import curve_fit
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None
        n_timepoints = self.experiments[0].get_file_time_points()
        all_ITs = np.empty((n_experiments, n_timepoints))
        peak_amplitude_positions = []

        actual_index = replicate_time_point
        for i, experiment in enumerate(self.experiments):
            file = experiment.get_spheroid_file(actual_index)
            IT_individual = file.get_processed_data_IT()
            if IT_individual.shape[0] != n_timepoints:
                raise ValueError(
                    f"Replicate {i+1} has {IT_individual.shape[0]} time points, expected {n_timepoints}.\n"
                    "All replicates must have the same number of time points."
                )
            metadata = file.get_metadata()
            peak_amplitude_positions.append((metadata["peak_amplitude_positions"]))
            all_ITs[i, :] = IT_individual

        # Turning peak_amplitude_positions into a list of integers
        try:
            peaks = [p.item() for p in peak_amplitude_positions]
        except AttributeError:
            peaks = [int(p) for p in peak_amplitude_positions]
        min_peak = np.min(peaks)
        max_peak = np.max(peaks)
        pre_allocated_ITs_array = np.full((n_experiments, n_timepoints - min_peak), np.nan)       
        # Fill the pre-allocated array with the cropped ITs, starting from the peak position
        for i, (row, peak) in enumerate(zip(all_ITs, peaks)):
            # Now the array will have from the min peak position to the end of the time points
            # Now this has aligned the peaks, so the first time point is the peak position for all ITs
            cropped = row[peak:]
            length = cropped.shape[0]
            pre_allocated_ITs_array[i, :length] = cropped

        # Crop the ITs from the end to match data sizes
        cropped_ITs = pre_allocated_ITs_array[:, :n_timepoints-max_peak-min_peak]
        print("Cropped ITs:",np.shape(cropped_ITs))
        ITs_flattened = cropped_ITs.flatten()
        print(np.shape(ITs_flattened))
        n_cropped_timepoints = np.shape(cropped_ITs)

        A = np.arange(min_peak, n_timepoints-max_peak)
        print(np.shape(A))
        n_post = n_timepoints - max_peak - min_peak
        A = np.arange(n_post)               # 0,1,2,... in samples
        time_all = np.tile(A, n_experiments)  # Repeat time point
        print(np.shape(time_all))

        mean_trace = np.mean(cropped_ITs, axis=0)
        C0 = np.median(mean_trace[-10:])
        A0 = np.mean(mean_trace[0:10])
        k0 = 0.01
        p0 = [k0]

        #Fit k only, fix A0 and C0
        def exp_decay_k_only(t, k):
            return (A0 - C0) * np.exp(-k * t) + C0

        popt_k, pcov_k = curve_fit(exp_decay_k_only, time_all, ITs_flattened, p0=[k0])
        k_fit = popt_k[0]
        k_err = np.sqrt(np.diag(pcov_k))[0]

        tau_fit = 1 / k_fit
        t_half = np.log(2) * tau_fit

        # Fit A only, fix k and C0
        def exp_decay_fixed_kC(t, A):
            return (A - C0) * np.exp(-k_fit * t) + C0

        popt_a, pcov_a = curve_fit(exp_decay_fixed_kC, time_all, ITs_flattened, p0=[A0])
        A_fit = popt_a[0]
        A_err = np.sqrt(np.diag(pcov_a))[0]

        # Fit C only, fix k and A
        def exp_decay_fixed_kA(t, C):
            return (A_fit - C) * np.exp(-k_fit * t) + C

        popt_c, pcov_c = curve_fit(exp_decay_fixed_kA, time_all, ITs_flattened, p0=[C0])
        C_fit = popt_c[0]
        C_err = np.sqrt(np.diag(pcov_c))[0]

        # Final report
        #print("Fit results (sequential):")
        #print(f"k = {k_fit:.4f} ± {k_err:.4f}")
        #print(f"A = {A_fit:.4f} ± {A_err:.4f}")
        #print(f"C = {C_fit:.4f} ± {C_err:.4f}")
        #print(f"Tau = {tau_fit:.4f}")
        #print(f"t_half = {t_half:.4f}")

        # Pre-allocated_ITs_array is the matrix with all data properly aligned on their peaks
        return time_all, cropped_ITs, pre_allocated_ITs_array, t_half, (A_fit, k_fit, C_fit), (A_err, k_err, C_err), min_peak
    
    def get_tau_over_time(self):
        """Extract tau (decay time constant) at each replicate time point.

        Returns:
            tuple: (tau_list, tau_error_list)
        """
        n_files = self.experiments[0].get_file_count()
        tau_list = []
        tau_err_list = []
        for t in range(n_files):
            try:
                _, _, _, t_half, fit_vals, fit_errs, _ = self.exponential_fitting_replicated(replicate_time_point=t)
                # fit_vals = (A_fit, k_fit, C_fit), fit_errs = (A_err, k_err, C_err)
                k_fit = fit_vals[1]
                k_err = fit_errs[1]
                tau_fit = 1 / k_fit if k_fit != 0 else np.nan
                tau_err = abs(k_err / (k_fit ** 2)) if k_fit != 0 else np.nan
                tau_list.append(tau_fit)
                tau_err_list.append(tau_err)
            except Exception as e:
                tau_list.append(np.nan)
                tau_err_list.append(np.nan)
        return tau_list, tau_err_list
    
    def get_exponential_fit_params_over_time(self):
        """
        Runs exponential_fitting_replicated for each replicate time point,
        collects A, tau, C, t_half and their errors, and returns them as a 2D numpy array.
        Columns: A_fit, A_error, tau_fit, tau_error, C_fit, C_error, t_half, t_half_error
        Rows: replicate time points
        The errors here are the standard error of the different quantities, in other words, 
        what is computed is the one-sigma (≈ 68 % coverage) standard error.

        Returns:
            np.ndarray: A matrix of shape (n_timepoints, 16) with values and uncertainties.
        """
        n_files = self.experiments[0].get_file_count()
        n_reps = len(self.experiments)
        z95 = 1.96

        results = []
        for t in range(n_files):
            try:
                _, _, _, t_half, fit_vals, fit_errs, _ = self.exponential_fitting_replicated(replicate_time_point=t)
                A_fit, k_fit, C_fit = fit_vals
                A_SE,  k_SE,   C_SE = fit_errs

                tau_fit   = 1.0 / k_fit
                tau_SE    = abs(k_SE   / k_fit**2)
                t_half    = np.log(2) * tau_fit
                t_half_SE = np.log(2) * tau_SE

                # convert SE to sample SD
                A_SD      = A_SE      * np.sqrt(n_reps)
                tau_SD    = tau_SE    * np.sqrt(n_reps)
                C_SD      = C_SE      * np.sqrt(n_reps)
                t_half_SD = t_half_SE * np.sqrt(n_reps)

                # 95% CI half‐widths
                A_CI95      = z95 * A_SE
                tau_CI95    = z95 * tau_SE
                C_CI95      = z95 * C_SE
                t_half_CI95 = z95 * t_half_SE

                results.append([
                    A_fit,   A_SE,   A_SD,   A_CI95,
                    tau_fit, tau_SE, tau_SD, tau_CI95,
                    C_fit,   C_SE,   C_SD,   C_CI95,
                    t_half,  t_half_SE, t_half_SD, t_half_CI95
                ])

            except Exception as e:
                results.append([np.nan]*16)
        return np.array(results)
    
    def exponential_fitting_replicated_legacy(self, replicate_time_point = 0, global_peak_amplitude_position=None):
        """
        Legacy Function - No Longer Supporter. This function performs exponential fitting of IT curves without aligning the peaks across replicates.

        Args:
            replicate_time_point (int): Index of the replicate time point (e.g., 3 for first post-treatment).
            global_peak_amplitude_position (int, optional): Optional override for peak alignment index.

        Returns:
            tuple: A tuple containing:
                - time_all (np.ndarray): Time vector repeated across replicates.
                - ITs_flattened (np.ndarray): Flattened IT signal array.
                - t_half (float): Estimated half-life of the decay.
                - popt (list): Fitted parameters [A, tau, C].
                - pcov (np.ndarray): Covariance matrix of the fit.
                - A_fit (float): Fitted amplitude.
                - tau_fit (float): Fitted decay constant.
                - C_fit (float): Fitted baseline value.
        """

        from scipy.optimize import curve_fit
        
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None

        # Assume all experiments have the same number of files/timepoints
        n_timepoints = self.experiments[0].get_file_time_points()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment() # This will be zero if no files before treatment
        
        all_ITs = np.empty((n_experiments, n_timepoints))
        peak_amplitude_positions = []

        actual_index = replicate_time_point + files_before_treatment
        for i, experiment in enumerate(self.experiments):
            file = experiment.get_spheroid_file(actual_index)
            IT_individual = file.get_processed_data_IT()
            metadata = file.get_metadata()
            peak_amplitude_positions.append(metadata["peak_amplitude_positions"])
            all_ITs[i, :] = IT_individual

        print(peak_amplitude_positions)
        latest_peak_amplitude_positions = np.max(peak_amplitude_positions)
        if global_peak_amplitude_position is None:
            global_peak_amplitude_position = latest_peak_amplitude_positions
        else:
            global_peak_amplitude_position = int(global_peak_amplitude_position)
        print(global_peak_amplitude_position)
        cropped_ITs = all_ITs[:, global_peak_amplitude_position:]

        ITs_flattened = cropped_ITs.flatten()
        n_cropped_timepoints = np.shape(cropped_ITs)

        A = np.arange(global_peak_amplitude_position, n_timepoints)
        time_all = np.tile(A, n_experiments)  # Repeat time point

        #print("Len ITs Flattened:", len(ITs_flattened))
        #print("ITs_Flattened", np.shape(ITs_flattened))
        #print("Time All", np.shape(time_all))

        # Improved initial guess for parameters
        # A: amplitude (difference between max and min of cropped ITs)
        # tau: decay constant (guess as 1/3 of the time range)
        # C: baseline (last value of the mean trace)
        mean_trace = np.mean(cropped_ITs, axis=0)
        A0 = float(np.max(mean_trace) - np.min(mean_trace))
        tau0 = (n_timepoints - global_peak_amplitude_position) / 3.0
        C0 = float(mean_trace[-1])
        p0 = [A0, tau0, C0]

        print(f"Initial guess: A={A0:.2f}, tau={tau0:.2f}, C={C0:.2f}")

        # Fit
        popt, pcov = curve_fit(exp_decay, time_all, ITs_flattened, p0=p0)

        # Extract parameter estimates and standard errors
        A_fit, tau_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))  # Approximate symmetric 1-sigma CI

        print(f"Fit results:")
        print(f"A   = {A_fit:.2f} ± {perr[0]:.2f}")
        print(f"tau = {tau_fit:.2f} ± {perr[1]:.2f}")
        print(f"C   = {C_fit:.2f} ± {perr[2]:.2f}")

        t_half = np.log(2) * tau_fit

        return time_all, ITs_flattened, t_half, popt, pcov,  A_fit, tau_fit, C_fit

    def plot_exponential_fit_aligned(self, replicate_time_point=0, save_path=None):
        """
        Plot the exponential decay fit on peak-aligned IT curves with a 95% confidence interval.

        Args:
            replicate_time_point (int): Index of the replicate time point to analyze.
            save_path (str, optional): File path to save the figure. If None, shows the plot.

        Returns:
            tuple: A tuple containing the matplotlib figure and axis objects.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import t
        import numpy as np

        # 1) run your fit and get the aligned, cropped IT matrix
        time_all, cropped_ITs, _, t_half, fit_vals, fit_errs, _ = \
            self.exponential_fitting_replicated(replicate_time_point)
        A_fit, k_fit, C_fit = fit_vals
        A_err, k_err, C_err = fit_errs
        tau_fit = 1 / k_fit if k_fit != 0 else np.nan

        # 2) build a “time since peak” axis
        n_exps, n_post = cropped_ITs.shape
        t_rel = np.arange(n_post)

        # 3) compute mean ± SD across replicates
        mean_IT = np.nanmean(cropped_ITs, axis=0)
        std_IT  = np.nanstd (cropped_ITs, axis=0)

        # 4) smooth fit curve on that same relative axis
        t_fit_rel = np.linspace(0, n_post-1, 500)
        y_fit     = A_fit * np.exp(-t_fit_rel * k_fit) + C_fit

        # 5) 95% CI of the fit via Jacobian
        dof  = max(0, len(time_all) - 3)
        tval = t.ppf(0.975, dof)
        J        = np.empty((len(t_fit_rel), 3))
        J[:, 0]  = np.exp(-t_fit_rel * k_fit)
        J[:, 1]  = -A_fit * t_fit_rel * np.exp(-t_fit_rel * k_fit)
        J[:, 2]  = 1
        pcov = np.diag([A_err**2, k_err**2, C_err**2])
        ci       = np.sqrt(np.sum((J @ pcov) * J, axis=1)) * tval
        lower_ci = y_fit - ci
        upper_ci = y_fit + ci

        # 6) start plotting
        fig, ax = plt.subplots(figsize=(10,6))

        for row in cropped_ITs:
            ax.plot(t_rel, row, color='gray', alpha=0.3, lw=1, label='_nolegend_')
        ax.fill_between(t_rel, mean_IT - std_IT, mean_IT + std_IT, color='C0', alpha=0.2, label='Mean ± 1 SD')
        ax.plot(t_rel, mean_IT, color='C0', lw=2, label='Mean trace')
        ax.plot(t_fit_rel, y_fit, color='C1', lw=2, label='Exp fit')
        ax.fill_between(t_fit_rel, lower_ci, upper_ci, color='C1', alpha=0.3, label='95% CI')
        ax.axvline(t_half, color='magenta', ls='--', label=f't½ ≈ {t_half:.1f} pts')
        ax.set_xlabel('Time since peak (points)', fontsize=12)
        ax.set_ylabel('Current (nA)', fontsize=12)
        ax.set_title('Post-peak IT decays & exponential fit', fontsize=14)
        ax.legend(frameon=False)
        ax.grid(False)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
        n_files = self.experiments[0].get_file_count()
        time_points = np.linspace(
            0,
            self.experiments[0].get_time_between_files() * (n_files - 1),
            n_files
        )

        plt.figure(figsize=(10, 6))
        plt.errorbar(time_points, tau_list, yerr=tau_err_list, fmt='o-', capsize=4, color='C1', label='Tau (decay constant)')
        plt.xlabel("Time (minutes)")
        plt.ylabel("Tau (decay constant)")
        plt.title("Exponential Decay Tau Over Time Points")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_exponential_fit_with_CI_legacy(self, replicate_time_point=0, global_peak_position=None):
        """
        Legacy Function - No Longer in Use, This functions plots an exponential fit and confidence interval over raw IT traces.

        Args:
            replicate_time_point (int): Index of the replicate time point.
            global_peak_position (int, optional): Override for alignment if specified.

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        from scipy.stats import t
        import numpy as np

        # Run the fitting function to get values and data
        time_all, ITs_flattened, t_half, popt, pcov, A_fit, tau_fit, C_fit = self.exponential_fitting_replicated_legacy(replicate_time_point=replicate_time_point)
        A_fit, tau_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))  # 1-sigma CI

        # Retrieve full ITs and metadata again for plotting individual traces
        n_experiments = len(self.experiments)
        file_duration = self.experiments[0].get_file_length()
        n_timepoints = self.experiments[0].get_file_time_points()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment()
        actual_index = replicate_time_point + files_before_treatment

        all_ITs = np.empty((n_experiments, n_timepoints))
        peak_positions = []

        for i, experiment in enumerate(self.experiments):
            file = experiment.get_spheroid_file(actual_index)
            IT_individual = file.get_processed_data_IT()
            metadata = file.get_metadata()
            peak_pos = metadata.get("peak_amplitude_positions")
            peak_positions.append(peak_pos)
            all_ITs[i, :] = IT_individual

        global_peak_position = int(np.max(peak_positions))
        if global_peak_position is None:
            global_peak_position = int(np.max(peak_positions))
        else:
            global_peak_position = global_peak_position
        # Time array for full profile
        full_time = np.arange(n_timepoints)

        # Time for fitting and prediction
        t_fit = np.linspace(global_peak_position, n_timepoints - 1, 500)
        y_fit = A_fit * np.exp(-t_fit / tau_fit) + C_fit

        # 95% CI using Jacobian
        dof = max(0, len(time_all) - len(popt))
        t_val = t.ppf(0.975, dof)

        J = np.empty((len(t_fit), 3))
        J[:, 0] = np.exp(-t_fit / tau_fit)
        J[:, 1] = A_fit * t_fit / tau_fit**2 * np.exp(-t_fit / tau_fit)
        J[:, 2] = 1

        ci = np.sqrt(np.sum((J @ pcov) * J, axis=1)) * t_val
        y_lower = y_fit - ci
        y_upper = y_fit + ci

        # Plot
        plt.figure(figsize=(10, 6))

        # Plot each replicate I-T trace
        for i in range(n_experiments):
            plt.plot(full_time, all_ITs[i, :], alpha=0.4, linewidth=1, label=f"Replicate {i+1}" if i == 0 else None)

        # Plot fit and CI
        plt.plot(t_fit, y_fit, label='Exponential Fit', color='red', linewidth=2)
        plt.fill_between(t_fit, y_lower, y_upper, color='red', alpha=0.3, label='95% CI')

        # Mark peak and t_half
        plt.axvline(global_peak_position, color='orange', linestyle='--', label=f"Peak @ {global_peak_position}")
        plt.axvline(global_peak_position + int(t_half), color='purple', linestyle=':', label=f"t_half ≈ {t_half:.2f}")

        # Plot peak positions of each replicate
        for i, peak_pos in enumerate(peak_positions):
            if isinstance(peak_pos, (np.ndarray, list)) and len(peak_pos) > 0:
                pos = int(np.max(peak_pos))
            else:
                pos = int(peak_pos)
            plt.scatter(pos, all_ITs[i, pos], color='red', marker='x', s=10, label='Peak Position' if i == 0 else None, zorder=5)

        plt.xlabel("Time Points (seconds)")
        # Set ticks at every 10 seconds => every 100 time points
        tick_locs = np.arange(0, 601, 100)       # 0, 100, 200, ..., 600
        tick_labels = [str(int(x / 10)) for x in tick_locs]  # convert to seconds: 0, 10, ..., 60
        plt.xticks(tick_locs, tick_labels, fontsize=10)
        plt.ylabel("Current (nA)")
        plt.title("Replicate I-T Profiles with Exponential Fit & CI")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

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

        # Compute treatment time in seconds
        experiment = self.experiments[experiment_index]
        treatment_time = files_before_treatment * experiment.get_time_between_files() # his will be zero if no files before treatment

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
        plt.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')

        # Optional scatter for emphasis
        plt.scatter(time_points, amplitudes, color='black', s=20, alpha=0.6)

        # Scientific styling
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.xticks(np.arange(0, max(time_points) + 1, 10), fontsize=10)
        plt.title('Amplitude Over Time Relative to Treatment', fontsize=14)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

        treatment_time = files_before_treatment * (time_points[1] - time_points[0])

        plt.figure(figsize=(10, 6))
        plt.plot(time_points, mean_amplitudes, label='Mean Amplitude', color='purple')
        plt.fill_between(time_points, mean_amplitudes - std_amplitudes, mean_amplitudes + std_amplitudes,
                         color='purple', alpha=0.2, label='SD')
        if files_before_treatment > 0:
            plt.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Amplitude')
        plt.title('Mean Amplitude Over Time (All Experiments)')
        plt.legend()
        plt.xticks(np.arange(0, max(time_points) + 1, 10), fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        treatment_time = files_before_treatment * (time_points[1] - time_points[0])

        plt.figure(figsize=(10, 6))
        for i, amplitudes in enumerate(all_amplitudes):
            plt.plot(time_points, amplitudes, label=f'Experiment {i+1}', alpha=0.7)
        if files_before_treatment > 0:
            plt.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')
        plt.xlabel('Time (min)')
        plt.ylabel('Amplitude')
        plt.title('Amplitudes Over Time (All Experiments)')
        plt.legend()
        plt.xticks(np.arange(0, max(time_points) + 1, 10), fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

        # Time points (columns in mean_ITs)
        time_points = np.arange(mean_ITs.shape[1])

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
        plt.xlabel("Time Points", fontsize=14)
        plt.ylabel("Mean IT (nA)", fontsize=14)
        plt.title(f"Mean IT Profiles Over Replicates", fontsize=16)
        plt.legend(fontsize=10, loc="upper right")
        plt.grid(False)

        # Show the plot
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
