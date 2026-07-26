from core.spheroid_experiment import SpheroidExperiment
from core.group_analysis import GroupAnalysis
import os
import re
import pandas as pd
import numpy as np
from scipy.stats import sem
from datetime import datetime
import json

class OutputManager:
    @staticmethod
    def _treatment_suffix():
        """Filename suffix built from the treatment saved in QSettings.

        Returns ``"_<treatment>"`` (path-safe) or ``""`` when no treatment is
        configured — the settings default is the literal string "None", which
        is treated as "not set" so filenames stay clean.
        """
        from PyQt5.QtCore import QSettings
        treatment = QSettings("HashemiLab", "NeuroStemVolt").value("treatment", "", type=str)
        treatment = (treatment or "").strip()
        if not treatment or treatment.lower() == "none":
            return ""
        # collapse anything that is not filename-safe into single underscores
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", treatment).strip("._-")
        return f"_{safe}" if safe else ""

    @staticmethod
    def _with_treatment(filename):
        """Insert the treatment suffix before the extension of ``filename``.

        e.g. ``exp_fit_joint.csv`` -> ``exp_fit_joint_Sertraline.csv``.
        """
        base, ext = os.path.splitext(filename)
        return f"{base}{OutputManager._treatment_suffix()}{ext}"

    @staticmethod
    def save_ITs(group_experiments : GroupAnalysis, output_folder_path):
        """
        Save processed IT data for each experiment into individual CSV files.

        Args:
            group_experiments (GroupAnalysis): The group containing experiments with IT data.
            output_folder_path (str): Directory where the CSV files will be saved.

        Returns:
            None
        """
        # This function takes all experiments (after processing) 
        # and creates output_csv files for all of them
        n_experiments = len(group_experiments.get_experiments())
        if n_experiments == 0:
            return None
        # Initialise the matrix 
        for i, experiment in enumerate(group_experiments.get_experiments()):
            it_matrix = []
            file_names = []
            for j, spheroid_file in enumerate(experiment.files):
                IT_individual = spheroid_file.get_processed_data_IT()
                it_matrix.append(IT_individual)
                file_name = spheroid_file.get_filepath()
                file_names.append(file_name)
            # Transpose so each column is a file
            df = pd.DataFrame(it_matrix).T
            df.columns = [f"File_{i}" for i in range(len(file_names))]
            df.columns = [file_name.split("/")[-1] for file_name in file_names]
            output_csv = OutputManager._with_treatment("All_ITs_experiment_n{0}.csv".format(i))
            output_IT_folder = os.path.join(output_folder_path,"replicate_ITs")
            if os.path.isdir(output_IT_folder) == False:
                os.mkdir(output_IT_folder)
            output_path = os.path.join(output_IT_folder, output_csv)
            df.to_csv(output_path, index_label="TimePoint")

    @staticmethod
    def save_all_ITs(group_experiments : GroupAnalysis, output_folder_path):
        """
        Save all IT traces from all replicates into a single multi-indexed CSV file.

        Args:
            group_experiments (GroupAnalysis): Group containing multiple experiments.
            output_folder_path (str): Directory to save the output file.

        Returns:
            None
        """
        # This function takes all experiments (after processing) 
        # and creates output_csv files for all of them
        experiments = group_experiments.get_experiments()
        n_experiments = len(group_experiments.get_experiments())
        n_ITs = group_experiments.get_experiments()[0].get_file_count()
        n_timepoints = group_experiments.get_experiments()[0].get_file_time_points()
        if n_experiments == 0:
            return None
        # Initialise the matrix
        arrays = []
        data = []
        # For each experiment (replicate)
        for exp_idx, experiment in enumerate(experiments):
            rep_name = f"Rep{exp_idx+1}"
            for file_idx, spheroid_file in enumerate(experiment.files):
                file_short = os.path.basename(spheroid_file.get_filepath())
                arrays.append((rep_name, file_short))    
                
        for t in range(n_timepoints):
            row = []
            for exp_idx, experiment in enumerate(experiments):
                for file_idx, spheroid_file in enumerate(experiment.files):
                    IT_individual = spheroid_file.get_processed_data_IT()
                    if t < len(IT_individual):
                        row.append(IT_individual[t])
                    else:
                        row.append(None)
            data.append(row)

        # Create time axis in seconds
        acq_freq = group_experiments.get_single_experiments(0).get_acquisition_frequency()
        time_seconds = np.arange(n_timepoints) / acq_freq

        # Create MultiIndex columns
        columns = pd.MultiIndex.from_tuples(arrays, names=["Replicate", "File"])
        df = pd.DataFrame(data, columns=columns)
        df.index = time_seconds
        df.index.name = "Time (s)"

        # Save to CSV
        output_IT_folder = os.path.join(output_folder_path, "all_replicates_ITs")
        os.makedirs(output_IT_folder, exist_ok=True)
        output_path = os.path.join(output_IT_folder,
                                   OutputManager._with_treatment("All_ITs_all_replicates.csv"))
        df.to_csv(output_path)
        print(f"Saved all ITs for all replicates to {output_path}")

        df_paired = df.swaplevel("Replicate","File", axis=1)   \
                      .sort_index(axis=1)                    

        paired_folder = os.path.join(output_folder_path,
                                     "all_replicates_ITs_paired")
        os.makedirs(paired_folder, exist_ok=True)
        paired_path = os.path.join(
            paired_folder,
            OutputManager._with_treatment("All_ITs_all_replicates_paired_by_file.csv"))
        df_paired.to_csv(paired_path)
        print(f"Saved paired ITs (same file side-by-side) to {paired_path}")
        
    @staticmethod
    def save_original_ITs(group_experiments : GroupAnalysis, output_folder_path):
        """
        Save the original (unprocessed) IT data for each replicate into CSV files.

        Args:
            group_experiments (GroupAnalysis): Group with experiments to export.
            output_folder_path (str): Path to the destination folder.

        Returns:
            None
        """
        # This function takes all experiments (after processing) 
        # and creates output_csv files for all of them
        n_experiments = len(group_experiments.get_experiments())
        if n_experiments == 0:
            return None
        # Initialise the matrix 
        for i, experiment in enumerate(group_experiments.get_experiments()):
            it_matrix = []
            file_names = []
            for j, spheroid_file in enumerate(experiment.files):
                IT_individual = spheroid_file.get_original_data_IT()
                it_matrix.append(IT_individual)
                file_name = spheroid_file.get_filepath()
                file_names.append(file_name)
            # Transpose so each column is a file
            df = pd.DataFrame(it_matrix).T
            df.columns = [f"File_{i}" for i in range(len(file_names))]
            df.columns = [file_name.split("/")[-1] for file_name in file_names]
            output_csv = OutputManager._with_treatment("Original_ITs_experiment_n{0}.csv".format(i))
            output_IT_folder = os.path.join(output_folder_path,"original_ITs_per_replicate")
            if os.path.isdir(output_IT_folder) == False:
                os.mkdir(output_IT_folder)
            output_path = os.path.join(output_IT_folder, output_csv)
            df.to_csv(output_path, index_label="TimePoint")

    @staticmethod
    def save_peak_amplitudes_metrics(group_experiments : GroupAnalysis, output_folder_path):
        """
        Save selected peak amplitude metrics (position and value) from each file.

        Args:
            group_experiments (GroupAnalysis): The experiments to extract metadata from.
            output_folder_path (str): Directory for saving metadata CSVs.

        Returns:
            None
        """
        # This function saves the following keys per file in a folder named metadata_files
        # Particularly these are the saves the method can save:
        # # Current metadata:
            # dict_keys(['peak_position', 'stim_start', 'stim_duration', 'stim_frequency', 
            # 'background_subtraction_region', 'baseline', 'experiment_first_peak', 'peak_amplitude_positions', 
            # 'peak_amplitude_values', 'exponential fitting parameters']) dict_values([257, 5.0, 2.0, 20, (0, 10), 
            # array([0.02988434]), np.float64(0.3822367999525832), array([74]), array([1.04633804]), 
            # {'A': np.float64(1.5705880537206165), 'tau': np.float64(119.00028560994886), 
            # 'C': np.float64(0.22708457061563378), 't_half': np.float64(82.48471245636428)}])
        # This method saves the following keys: keys = ['peak_amplitude_positions','peak_amplitude_values']
        keys = ['peak_amplitude_positions','peak_amplitude_values']
        for i, experiment in enumerate(group_experiments.get_experiments()):
            records = []
            for j, spheroid_file in enumerate(experiment.files):
                meta = spheroid_file.get_metadata()
                if keys is None:
                    # Save all keys
                    records.append(meta)
                else:
                    # Save only selected keys
                    records.append({k: meta.get(k, None) for k in keys})
                df = pd.DataFrame(records)
                output_csv = OutputManager._with_treatment(
                    "Files_Amplitudes_experiment_n{0}.csv".format(i))
                output_IT_folder = os.path.join(output_folder_path,"amplitudes_files")
                if os.path.isdir(output_IT_folder) == False:
                    os.mkdir(output_IT_folder)
                output_path = os.path.join(output_IT_folder, output_csv)
                df.to_csv(output_path, index_label="File Number")

    @staticmethod
    def save_spontaneous_peak_metrics(group_experiments : GroupAnalysis, output_folder_path):
        """Save detailed metrics about spontaneous peaks for all experiments."""
        import pandas as pd
        import numpy as np
        import os
        from PyQt5.QtCore import QSettings

        settings = QSettings("HashemiLab", "NeuroStemVolt")
        acquisition_freq = settings.value("acquisition_frequency", 10, type=int)
        file_length_sec = settings.value("file_length", 100, type=int)
        time_between_files = settings.value("time_between_files", 10, type=float)

        # Create output folder if it doesn't exist
        spont_folder = os.path.join(output_folder_path, "spontaneous_metrics")
        if not os.path.isdir(spont_folder):
            os.mkdir(spont_folder)

        # Summary data for all experiments
        all_summary_records = []

        # Process each experiment
        for i, experiment in enumerate(group_experiments.get_experiments()):
            # Detailed records for each peak in this experiment
            detailed_records = []
            # Summary record for each file in this experiment
            summary_records = []

            for j, spheroid_file in enumerate(experiment.files):
                meta = spheroid_file.get_metadata()
                file_name = os.path.basename(spheroid_file.get_filepath())

                # Get peak data
                peak_positions = meta.get('peak_amplitude_positions', [])
                peak_values = meta.get('peak_amplitude_values', [])
                peak_metadata = meta.get('all_peak_metadata', [])

                # Handle both single value and list/array cases
                if not isinstance(peak_positions, (list, np.ndarray)):
                    peak_positions = [peak_positions] if peak_positions else []

                if not isinstance(peak_values, (list, np.ndarray)):
                    peak_values = [peak_values] if peak_values else []

                # Calculate summary metrics
                num_peaks = len(peak_positions)
                mean_amplitude = np.mean(peak_values) if peak_values else 0
                peak_frequency = num_peaks / (file_length_sec / 60)  # peaks per minute
                timepoint_min = j * time_between_files

                # Add summary record
                summary_records.append({
                    'File Number': j,
                    'File Name': file_name,
                    'Time (min)': timepoint_min,
                    'Number of Peaks': num_peaks,
                    'Mean Amplitude (nA)': mean_amplitude,
                    'Peak Frequency (peaks/min)': peak_frequency
                })

                # Add detailed records for each peak
                for k, (pos, val) in enumerate(zip(peak_positions, peak_values)):
                    peak_info = {
                        'File Number': j,
                        'File Name': file_name,
                        'Time (min)': timepoint_min,
                        'Peak Number': k + 1,
                        'Peak Position': pos,
                        'Peak Time (s)': pos / acquisition_freq,
                        'Amplitude (nA)': val
                    }

                    # Add rise/decay info if available
                    if k < len(peak_metadata):
                        md = peak_metadata[k]
                        peak_info.update({
                            'Rise Time (s)': md.get('rise_time_sec', 0),
                            'Decay Time (s)': md.get('decay_time_sec', 0)
                        })

                    detailed_records.append(peak_info)

            # Save detailed peak information for this experiment
            if detailed_records:
                df_detailed = pd.DataFrame(detailed_records)
                output_detailed = OutputManager._with_treatment(
                    f"Experiment_{i+1}_Detailed_Peak_Data.csv")
                df_detailed.to_csv(os.path.join(spont_folder, output_detailed), index=False)

            # Save summary for this experiment
            if summary_records:
                df_summary = pd.DataFrame(summary_records)
                output_summary = OutputManager._with_treatment(
                    f"Experiment_{i+1}_Summary_Peak_Data.csv")
                df_summary.to_csv(os.path.join(spont_folder, output_summary), index=False)

                # Add to all experiments summary
                for record in summary_records:
                    record['Experiment'] = i + 1
                    all_summary_records.append(record)

        # Save combined summary for all experiments
        if all_summary_records:
            df_all = pd.DataFrame(all_summary_records)
            output_all = OutputManager._with_treatment(
                "All_Experiments_Spontaneous_Peak_Summary.csv")
            df_all.to_csv(os.path.join(spont_folder, output_all), index=False)

        # Calculate group statistics and save them
        if all_summary_records:
            df_all = pd.DataFrame(all_summary_records)

            # Group by timepoint
            timepoint_stats = []
            for time, group in df_all.groupby('Time (min)'):
                timepoint_stats.append({
                    'Time (min)': time,
                    'Mean Peak Frequency (peaks/min)': group['Peak Frequency (peaks/min)'].mean(),
                    'StdDev Peak Frequency': group['Peak Frequency (peaks/min)'].std(),
                    'Mean Amplitude (nA)': group['Mean Amplitude (nA)'].mean(),
                    'StdDev Amplitude': group['Mean Amplitude (nA)'].std(),
                    'Total Peaks': group['Number of Peaks'].sum(),
                    'Number of Experiments': len(group)
                })

            # Save timepoint statistics
            if timepoint_stats:
                df_timepoints = pd.DataFrame(timepoint_stats)
                output_timepoints = OutputManager._with_treatment(
                    "Group_Statistics_By_Timepoint.csv")
                df_timepoints.to_csv(os.path.join(spont_folder, output_timepoints), index=False)

        return spont_folder
    @staticmethod
    def save_all_peak_amplitudes(group_experiments : GroupAnalysis, output_folder_path):
        """
        Save all peak amplitude values and their positions (in seconds) for each replicate.

        Args:
            group_experiments (GroupAnalysis): The experiment group to analyze.
            output_folder_path (str): Directory to store the output.

        Returns:
            None
        """
        keys = ['peak_amplitude_values', 'peak_amplitude_positions']

        experiments = group_experiments.get_experiments()
        acq_freq = experiments[0].get_acquisition_frequency()
        n_experiments = len(experiments)
        n_files = experiments[0].get_file_count()
        n_before = experiments[0].get_number_of_files_before_treatment()
        interval = experiments[0].get_time_between_files()  # e.g., 10

        if n_experiments == 0:
            return None
        # Initialise the time axis (first column)
        if n_before > 0:
            time_points = [interval * (i - n_before) for i in range(n_files)]
        else:
            time_points = [i * interval for i in range(n_files)]

        all_amplitudes = []
        all_amplitude_pos = []
        for i, experiment in enumerate(group_experiments.get_experiments()):
            records_amp = []
            records_pos = []
            for j, spheroid_file in enumerate(experiment.files):
                meta = spheroid_file.get_metadata()
                # Save only selected keys
                records_amp.append(meta.get(keys[0], None) if keys else meta)
                records_pos.append(meta.get(keys[1], None) if keys else meta)
            all_amplitudes.append(records_amp)
            all_amplitude_pos.append(records_pos)
        
        # Build DataFrame
        df_amp = pd.DataFrame(all_amplitudes).T  # shape: (n_files, n_experiments)
        df_pos = pd.DataFrame(all_amplitude_pos).T / acq_freq

        df_amp.columns = [f"Rep{idx+1}" for idx in range(n_experiments)]
        df_pos.columns = [f"Rep{idx+1}" for idx in range(n_experiments)]

        df = pd.concat({'Amplitude': df_amp, 'Position (s)': df_pos}, axis=1)
        df.insert(0, "Time", time_points)

        # Save to CSV
        output_folder = os.path.join(output_folder_path, "all_replicates_amplitudes")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder,
                                   OutputManager._with_treatment("All_amplitudes_all_replicates.csv"))
        df.to_csv(output_path, index=False)
        print(f"Saved all amplitudes for all replicates to {output_path}")

    def save_all_AUC(group_experiments:GroupAnalysis, output_folder_path):
        """
        Save the area under the curve (AUC) for all replicates and files over time.

        Args:
            group_experiments (GroupAnalysis): Group containing processed experiments.
            output_folder_path (str): Directory where outputs will be saved.

        Returns:
            None
        """
        experiments = group_experiments.get_experiments()
        n_experiments = len(experiments)
        n_files = experiments[0].get_file_count()
        n_before = experiments[0].get_number_of_files_before_treatment()
        interval = experiments[0].get_time_between_files()  # e.g., 10

        if n_experiments == 0:
            return None
        # Initialise the time axis (first column)
        if n_before > 0:
            time_points = [interval * (i - n_before) for i in range(n_files)]
        else:
            time_points = [i * interval for i in range(n_files)]

        all_AUC = group_experiments.get_all_AUC()
        df_AUC = pd.DataFrame(all_AUC).T

        df_AUC.columns = [f"Rep{idx+1}" for idx in range(n_experiments)]
        df_AUC.insert(0, "Time", time_points)

        # Save to CSV
        output_folder = os.path.join(output_folder_path, "all_replicates_AUC")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder,
                                   OutputManager._with_treatment("All_AUC_all_replicates.csv"))
        df_AUC.to_csv(output_path, index=False)
        print(f"Saved all amplitudes for all replicates to {output_path}")

        # Compute mean and SEM (standard error of the mean)
        values_only = df_AUC.iloc[:, 1:].to_numpy()  # exclude "Time" column
        mean_auc = np.nanmean(values_only, axis=1)
        sem_auc = sem(values_only, axis=1, nan_policy='omit')

        df_mean_sem = pd.DataFrame({
            "Time": time_points,
            "Mean AUC": mean_auc,
            "SEM AUC": sem_auc
        })

        # Save mean and SEM
        mean_output_path = os.path.join(output_folder,
                                        OutputManager._with_treatment("Mean_AUC_with_SEM.csv"))
        df_mean_sem.to_csv(mean_output_path, index=False)
        print(f"Saved mean AUC with SEM to {mean_output_path}")

    @staticmethod
    def save_all_reuptake_curves(group_experiments : GroupAnalysis, output_folder_path):
        """
        Save aligned reuptake curves (post-peak ITs) into a multi-indexed CSV file.

        Args:
            group_experiments (GroupAnalysis): The group of processed experiments.
            output_folder_path (str): Directory for storing output.

        Returns:
            None
        """
        # This function takes all experiments (after processing) 
        # and creates output_csv files for all of them
        experiments = group_experiments.get_experiments()
        n_experiments = len(group_experiments.get_experiments())
        n_ITs = group_experiments.get_experiments()[0].get_file_count()
        n_timepoints = group_experiments.get_experiments()[0].get_file_time_points()
        if n_experiments == 0:
            return None
        
        curves = group_experiments.get_all_reuptake_curves()
        curves_aligned = curves.T # Our structure of the matrix is like save_all_ITs

        # Create time axis in seconds
        acq_freq = group_experiments.get_single_experiments(0).get_acquisition_frequency()
        time_seconds = np.arange(curves_aligned.shape[0]) / acq_freq

        # Initialise the matrix
        arrays = []
        data = []
        # For each experiment (replicate)
        for exp_idx, experiment in enumerate(experiments):
            rep_name = f"Rep{exp_idx+1}"
            for file_idx, spheroid_file in enumerate(experiment.files):
                file_short = os.path.basename(spheroid_file.get_filepath())
                arrays.append((rep_name, file_short))    

        # Create MultiIndex columns
        columns = pd.MultiIndex.from_tuples(arrays, names=["Replicate", "File"])
        df = pd.DataFrame(curves_aligned, columns=columns)
        df.index = time_seconds
        df.index.name = "Time (s)"

        # Save to CSV
        output_IT_folder = os.path.join(output_folder_path, "all_reuptakes")
        os.makedirs(output_IT_folder, exist_ok=True)
        output_path = os.path.join(output_IT_folder,
                                   OutputManager._with_treatment("All_reuptakes.csv"))
        df.to_csv(output_path)
        print(f"Saved all reuptakes for all replicates to {output_path}")

        df_paired = df.swaplevel("Replicate","File", axis=1)   \
                      .sort_index(axis=1)                    

        paired_folder = os.path.join(output_folder_path,
                                     "all_reuptakes_paired")
        os.makedirs(paired_folder, exist_ok=True)
        paired_path = os.path.join(
            paired_folder,
            OutputManager._with_treatment("All_reuptakes_paired_by_file.csv"))
        df_paired.to_csv(paired_path)
        print(f"Saved paired ITs (same file side-by-side) to {paired_path}")

    @staticmethod
    def save_all_exponential_fitting_params(group_experiments : GroupAnalysis, output_folder_path):
        """
        Save exponential fitting parameters (A, tau, C) and related statistics over time.

        Args:
            group_experiments (GroupAnalysis): Group with time-series IT data.
            output_folder_path (str): Directory to write parameter outputs.

        Returns:
            None
        """
        params_matrix = group_experiments.get_exponential_fit_params_over_time()
        experiments = group_experiments.get_experiments()
        freq = experiments[0].get_acquisition_frequency()
        n_experiments = len(experiments)
        n_files = experiments[0].get_file_count()
        n_before = experiments[0].get_number_of_files_before_treatment()
        interval = experiments[0].get_time_between_files()  # e.g., 10

        if n_experiments == 0:
            return None
        # Initialise the time axis (first column)
        if n_before > 0:
            time_points = [interval * (i - n_before) for i in range(n_files)]
        else:
            time_points = [i * interval for i in range(n_files)]
        
        # Build DataFrame with unpacked columns
        df = pd.DataFrame(params_matrix, columns=["A_fit",   "A_SE",   "A_SD",   "A_CI95",
                    "tau_fit", "tau_SE", "tau_SD", "tau_CI95",
                    "C_fit",   "C_SE",   "C_SD",   "C_CI95",
                    "t_half",  "t_half_SE", "t_half_SD", "t_half_CI95"])
        
        #df["Y0"] = df["A_fit"] + df["C_fit"]
        #df["Y0_SE"] = np.sqrt(df["A_SE"]**2 + df["C_SE"]**2)
        #df["Y0_SD"] = np.sqrt(df["A_SD"]**2 + df["C_SD"]**2)
        #df["Y0_CI95"] = np.sqrt(df["A_CI95"]**2 + df["C_CI95"]**2)
        df.insert(0, "Time", time_points)

        # Document the sample size behind each row's pooled statistics: Path A
        # pools every replicate, so n_used = n_experiments on a successful fit
        # (0 where the fit failed and the row is NaN).
        df.insert(1, "n_used", np.where(
            np.isfinite(df["tau_fit"].to_numpy(dtype=float)), n_experiments, 0))

        #y0_cols = ["Y0", "Y0_SE", "Y0_SD", "Y0_CI95"]
        #new_order = ["Time"] + y0_cols + [c for c in df.columns if c not in (["Time"] + y0_cols)]
        #df = df[new_order]

        df["tau_fit"]    = df["tau_fit"]    / freq
        df["tau_SE"]     = df["tau_SE"]     / freq
        df["tau_SD"]     = df["tau_SD"]     / freq
        df["tau_CI95"]   = df["tau_CI95"]   / freq

        df["t_half"]     = df["t_half"]     / freq
        df["t_half_SE"]  = df["t_half_SE"]  / freq
        df["t_half_SD"]  = df["t_half_SD"]  / freq
        df["t_half_CI95"]= df["t_half_CI95"]/ freq

        # Save to CSV — folder name describes the method (per-replicate fit, then pool stats)
        output_folder = os.path.join(output_folder_path, "exp_fit_per_replicate_then_pooled_stats")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(
            output_folder,
            OutputManager._with_treatment("exp_fit_per_replicate_then_pooled_stats.csv"))
        df.to_csv(output_path, index=False)
        print(f"Saved per-replicate-then-pooled exp-fit params to {output_path}")

    @staticmethod
    def save_all_exponential_fitting_params_global(group_experiments: GroupAnalysis, output_folder_path):
        """
        Save exponential fitting parameters from the GLOBAL method: all replicate
        post-peak IT traces for a given file index are pooled into one dataset and a
        SINGLE exponential is fit to that pooled data.

        This contrasts with `save_all_exponential_fitting_params`, which fits each
        replicate independently and then pools the resulting parameters. Use this
        method when you want a single best-fit kinetic per timepoint that weighs every
        data point from every replicate equally. Standard errors come from the
        covariance matrix of the single fit; 95% CI uses z = 1.96.

        Args:
            group_experiments (GroupAnalysis): Group with time-series IT data.
            output_folder_path (str): Directory to write parameter outputs.

        Returns:
            None
        """
        experiments = group_experiments.get_experiments()
        if not experiments:
            return None

        freq = experiments[0].get_acquisition_frequency()
        n_files = experiments[0].get_file_count()
        n_before = experiments[0].get_number_of_files_before_treatment()
        interval = experiments[0].get_time_between_files()

        if n_before > 0:
            time_points = [interval * (i - n_before) for i in range(n_files)]
        else:
            time_points = [i * interval for i in range(n_files)]

        z95 = 1.96
        n_cols = 12
        n_reps = len(experiments)   # the global fit pools every replicate
        rows = []
        n_used_col = []

        for t in range(n_files):
            rel_t = t - n_before
            try:
                result = group_experiments.exponential_fitting_replicated_legacy(
                    replicate_time_point=rel_t
                )
                if result is None or not isinstance(result, tuple) or len(result) < 8:
                    raise ValueError("legacy fit returned no result")

                _, _, t_half, _, pcov, A_fit, tau_fit, C_fit = result

                pcov = np.asarray(pcov, dtype=float)
                if pcov.shape != (3, 3) or not np.all(np.isfinite(pcov)):
                    raise ValueError("invalid covariance matrix")
                perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
                A_SE, tau_SE, C_SE = float(perr[0]), float(perr[1]), float(perr[2])
                t_half_SE = float(abs(np.log(2) * tau_SE)) if np.isfinite(tau_SE) else np.nan

                rows.append([
                    float(A_fit),   A_SE,     z95 * A_SE,
                    float(tau_fit), tau_SE,   z95 * tau_SE,
                    float(C_fit),   C_SE,     z95 * C_SE,
                    float(t_half),  t_half_SE, z95 * t_half_SE,
                ])
                n_used_col.append(n_reps)
            except Exception as e:
                print(f"[global exp-fit] file index {t}: skipping ({e})")
                rows.append([np.nan] * n_cols)
                n_used_col.append(0)   # fit failed -> nothing pooled

        df = pd.DataFrame(rows, columns=[
            "A_fit",    "A_SE",     "A_CI95",
            "tau_fit",  "tau_SE",   "tau_CI95",
            "C_fit",    "C_SE",     "C_CI95",
            "t_half",   "t_half_SE", "t_half_CI95",
        ])
        df.insert(0, "Time", time_points)
        df.insert(1, "n_used", n_used_col)   # sample size behind each pooled row

        # tau and t_half are in samples — convert to seconds
        for col in ("tau_fit", "tau_SE", "tau_CI95", "t_half", "t_half_SE", "t_half_CI95"):
            df[col] = df[col] / freq

        output_folder = os.path.join(output_folder_path, "exp_fit_global_to_all_replicates")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(
            output_folder,
            OutputManager._with_treatment("exp_fit_global_to_all_replicates.csv"))
        df.to_csv(output_path, index=False)
        print(f"Saved global (single-fit-to-all-replicates) exp-fit params to {output_path}")

    # ------------------------------------------------------------------
    #  Savers for the three new replicate-fitting methods.
    #  Shared conventions vs. the older savers:
    #   * Units are LABELLED in the headers (Time_min vs. *_s in seconds).
    #   * No `_SD = SE*sqrt(n)` columns — that identity is invalid here.
    #     (Two-stage instead writes the real std(tau_i) as tau_SD_s.)
    #   * A `method` tag and per-timepoint `n_used` column are always present.
    #   * Any caught fit exception is PRINTED with its timepoint and reason,
    #     never silently swallowed.
    # ------------------------------------------------------------------
    @staticmethod
    def _timepoints_minutes(experiments):
        """Per-file time axis in minutes (baseline files given negative times)."""
        n_files = experiments[0].get_file_count()
        n_before = experiments[0].get_number_of_files_before_treatment()
        interval = experiments[0].get_time_between_files()
        if n_before > 0:
            return [interval * (i - n_before) for i in range(n_files)]
        return [i * interval for i in range(n_files)]

    @staticmethod
    def _common_fit_row(time_min, result, freq):
        """Seconds-converted common columns shared by all three method CSVs.

        tau/t_half sample quantities are divided by ``freq`` for value, SE and
        both CI endpoints alike; k is reported per second (k_samples * freq).
        """
        k = result.get("k", np.nan)
        se_k = result.get("se_k", np.nan)
        tau_ci = result.get("tau_ci", (np.nan, np.nan))
        th_ci = result.get("t_half_ci", (np.nan, np.nan))
        return {
            "Time_min": time_min,
            "method": result.get("method", ""),
            "n_used": result.get("n_used", 0),
            "k_fit_persec": k * freq if np.isfinite(k) else np.nan,
            "k_SE_persec": se_k * freq if np.isfinite(se_k) else np.nan,
            "tau_fit_s": result.get("tau", np.nan) / freq,
            "tau_SE_s": result.get("se_tau", np.nan) / freq,
            "tau_CI95_lo_s": tau_ci[0] / freq,
            "tau_CI95_hi_s": tau_ci[1] / freq,
            "t_half_s": result.get("t_half", np.nan) / freq,
            "t_half_SE_s": result.get("se_t_half", np.nan) / freq,
            "t_half_CI95_lo_s": th_ci[0] / freq,
            "t_half_CI95_hi_s": th_ci[1] / freq,
        }

    @staticmethod
    def _save_per_replicate_AC(rows, output_folder, filename, tag):
        """Write a timepoint x replicate table of per-replicate A_i and C_i."""
        if not rows:
            return
        df = pd.DataFrame(rows)
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, filename)
        df.to_csv(path, index=False)
        print(f"Saved per-replicate A_i/C_i ({tag}) to {path}")

    @staticmethod
    def save_exp_fit_joint(group_experiments: GroupAnalysis, output_folder_path):
        """Method 1 (adjusted Path A): pooled simultaneous (A, k, C) fit per timepoint.

        The pooled A and C are single shared values here (not per-replicate as in
        methods 2 and 3), so they go in the main CSV instead of a separate
        per-replicate file. A is the model value at t=0 and C the plateau, both
        in the units of the signal (nA or nM); their CIs are the symmetric
        ``estimate +/- t*SE`` using the same Student-t multiplier as tau.
        """
        experiments = group_experiments.get_experiments()
        if len(experiments) == 0:      # guard BEFORE indexing experiments[0]
            return None
        freq = experiments[0].get_acquisition_frequency()
        n_files = experiments[0].get_file_count()
        time_points = OutputManager._timepoints_minutes(experiments)

        rows = []
        for t in range(n_files):
            try:
                result = group_experiments.exponential_fitting_joint(replicate_time_point=t)
                row = OutputManager._common_fit_row(time_points[t], result, freq)
                # pooled amplitude/offset of the same 3-parameter fit (signal units)
                A = result.get("A", np.nan)
                A_se = result.get("A_se", np.nan)
                tmult = result.get("tmult", 1.96)
                row["A_fit"] = A
                row["A_SE"] = A_se
                row["A_CI95_lo"] = A - tmult * A_se
                row["A_CI95_hi"] = A + tmult * A_se
                C = result.get("C", np.nan)
                C_se = result.get("C_se", np.nan)
                row["C_fit"] = C
                row["C_SE"] = C_se
                row["C_CI95_lo"] = C - tmult * C_se
                row["C_CI95_hi"] = C + tmult * C_se
                row["status"] = "ok"
                rows.append(row)
            except Exception as e:
                print(f"[joint exp-fit] timepoint {t} (t={time_points[t]} min): skipping ({e})")
                rows.append({"Time_min": time_points[t], "method": "joint",
                             "n_used": 0, "status": f"failed: {e}"})

        df = pd.DataFrame(rows)
        # Fixed layout: one (value, SE, CI lo, CI hi) block per parameter in
        # A, C, k, tau, t_half order, then status, with n_used last. reindex
        # also keeps the header stable when a timepoint failed and only the
        # error keys were written for it.
        tail = ["status", "n_used"]
        head = ["Time_min", "method",
                "A_fit", "A_SE", "A_CI95_lo", "A_CI95_hi",
                "C_fit", "C_SE", "C_CI95_lo", "C_CI95_hi",
                "k_fit_persec", "k_SE_persec",
                "tau_fit_s", "tau_SE_s", "tau_CI95_lo_s", "tau_CI95_hi_s",
                "t_half_s", "t_half_SE_s", "t_half_CI95_lo_s", "t_half_CI95_hi_s"]
        extras = [c for c in df.columns if c not in head + tail]
        df = df.reindex(columns=head + extras + tail)

        output_folder = os.path.join(output_folder_path, "exp_fit_joint")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder,
                                   OutputManager._with_treatment("exp_fit_joint.csv"))
        df.to_csv(output_path, index=False)
        print(f"Saved joint (simultaneous A,k,C) exp-fit params to {output_path}")

    @staticmethod
    def save_exp_fit_shared_k(group_experiments: GroupAnalysis, output_folder_path):
        """Method 2 (shared-k fixed effects): one k, per-replicate A_i/C_i per timepoint.

        Main CSV holds the shared-k results (+ profile-likelihood CI); the
        per-replicate A_i and C_i go to a SEPARATE timepoint x replicate file.
        """
        experiments = group_experiments.get_experiments()
        if len(experiments) == 0:      # guard BEFORE indexing experiments[0]
            return None
        freq = experiments[0].get_acquisition_frequency()
        n_files = experiments[0].get_file_count()
        n_reps = len(experiments)
        time_points = OutputManager._timepoints_minutes(experiments)

        rows = []
        per_rep_rows = []
        for t in range(n_files):
            try:
                result = group_experiments.exponential_fitting_shared_k(replicate_time_point=t)
                row = OutputManager._common_fit_row(time_points[t], result, freq)
                # profile-likelihood interval on tau (seconds), if computed
                tau_prof = result.get("tau_prof_ci", (np.nan, np.nan))
                row["tau_profCI_lo_s"] = tau_prof[0] / freq
                row["tau_profCI_hi_s"] = tau_prof[1] / freq
                row["status"] = "ok"
                rows.append(row)

                A_i = np.asarray(result["A"], dtype=float)
                C_i = np.asarray(result["C"], dtype=float)
                pr = {"Time_min": time_points[t]}
                for r in range(n_reps):
                    pr[f"A_Rep{r+1}"] = A_i[r] if r < len(A_i) else np.nan
                    pr[f"C_Rep{r+1}"] = C_i[r] if r < len(C_i) else np.nan
                per_rep_rows.append(pr)
            except Exception as e:
                print(f"[shared-k exp-fit] timepoint {t} (t={time_points[t]} min): skipping ({e})")
                rows.append({"Time_min": time_points[t], "method": "shared_k",
                             "n_used": 0, "status": f"failed: {e}"})

        output_folder = os.path.join(output_folder_path, "exp_fit_shared_k")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder,
                                   OutputManager._with_treatment("exp_fit_shared_k.csv"))
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved shared-k (fixed-effects) exp-fit params to {output_path}")
        OutputManager._save_per_replicate_AC(
            per_rep_rows, output_folder,
            OutputManager._with_treatment("exp_fit_shared_k_per_replicate_AC.csv"), "shared_k")

    @staticmethod
    def save_exp_fit_two_stage(group_experiments: GroupAnalysis, output_folder_path):
        """Method 3 (two-stage): per-replicate fits combined linearly (mean +/- t*SD/sqrt(n)).

        Main CSV holds tau_bar with the between-replicate SE/CI, the real
        std(tau_i) of the KEPT fits, the two-tier drop accounting (stage-1
        non-fits and stage-2 SD outliers), and the variance diagnostic.
        Per-replicate A_i/C_i go to a SEPARATE timepoint x replicate file.
        """
        experiments = group_experiments.get_experiments()
        if len(experiments) == 0:      # guard BEFORE indexing experiments[0]
            return None
        freq = experiments[0].get_acquisition_frequency()
        n_files = experiments[0].get_file_count()
        n_reps = len(experiments)
        time_points = OutputManager._timepoints_minutes(experiments)

        rows = []
        per_rep_rows = []
        for t in range(n_files):
            try:
                result = group_experiments.exponential_fitting_two_stage(
                    replicate_time_point=t, verbose=True)
                row = OutputManager._common_fit_row(time_points[t], result, freq)
                # real between-replicate SD of the KEPT tau_i (NOT SE*sqrt(n))
                row["tau_SD_s"] = result.get("std_tau", np.nan) / freq
                # stage-1 non-fit rejections
                rej = result.get("rejected", [])
                row["n_valid"] = result.get("n_valid", np.nan)
                row["n_rejected"] = len(rej)
                row["rejected_replicates"] = ";".join(str(i + 1) for i, _c, _m in rej)
                row["rejection_reasons"] = ";".join(cat for _i, cat, _m in rej)
                # stage-2 SD outliers (excluded from the mean/CI)
                outs = result.get("outliers", [])
                row["n_outliers"] = len(outs)
                row["outlier_replicates"] = ";".join(str(i + 1) for i, _tau in outs)
                # variance diagnostic (linear, seconds)
                diag = group_experiments.replicate_variance_diagnostic(replicate_time_point=t)
                if diag is not None:
                    row["diag_s_s"] = diag["s"] / freq
                    row["diag_se_bar_s"] = diag["se_bar"] / freq
                    row["diag_sigma_between_s"] = diag["sigma_b_hat"] / freq
                row["status"] = result.get("status", "ok")
                rows.append(row)

                A_i = np.asarray(result["A"], dtype=float)
                C_i = np.asarray(result["C"], dtype=float)
                used = result.get("used_idx", list(range(len(A_i))))
                pr = {"Time_min": time_points[t]}
                for r in range(n_reps):
                    if r in used:
                        j = used.index(r)
                        pr[f"A_Rep{r+1}"] = A_i[j]
                        pr[f"C_Rep{r+1}"] = C_i[j]
                    else:
                        pr[f"A_Rep{r+1}"] = np.nan   # replicate dropped at this timepoint
                        pr[f"C_Rep{r+1}"] = np.nan
                per_rep_rows.append(pr)
            except Exception as e:
                print(f"[two-stage exp-fit] timepoint {t} (t={time_points[t]} min): skipping ({e})")
                rows.append({"Time_min": time_points[t], "method": "two_stage",
                             "n_used": 0, "status": f"failed: {e}"})

        output_folder = os.path.join(output_folder_path, "exp_fit_two_stage")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder,
                                   OutputManager._with_treatment("exp_fit_two_stage.csv"))
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved two-stage (per-replicate, linear mean +/- t*SD/sqrt(n)) exp-fit params to {output_path}")
        OutputManager._save_per_replicate_AC(
            per_rep_rows, output_folder,
            OutputManager._with_treatment("exp_fit_two_stage_per_replicate_AC.csv"), "two_stage")

    ### Methods for spheroid_files
    @staticmethod
    def save_IT_profile(spheroid_file, output_path):
        """
        Save the processed IT profile of a single file to a CSV.

        Args:
            spheroid_file: A SpheroidFile instance containing processed IT data.
            output_path (str): Directory where the IT CSV will be saved.

        Returns:
            np.ndarray: The processed IT data that was saved.
        """
        processed_data_IT = spheroid_file.get_processed_data_IT()

        n_timepoints = spheroid_file.timeframe
        # Create time axis in seconds
        acq_freq = spheroid_file.acq_freq

        print("n_timepoints:", n_timepoints,      "→", type(n_timepoints))
        print("acq_freq:   ", acq_freq,           "→", type(acq_freq))
        time_seconds = np.arange(float(n_timepoints)) / float(acq_freq)

        df = pd.DataFrame(processed_data_IT)
        df.index = time_seconds
        df.index.name = "Time (s)"

        base_name = os.path.splitext(os.path.basename(spheroid_file.get_filepath()))[0]  # Remove .txt
        df.columns = [base_name]
        output_file_name = OutputManager._with_treatment(base_name + "_IT.csv")
        
        # Save to CSV
        output_folder = os.path.join(output_path, "ITs")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_file_name)
        df.to_csv(output_path, index_label="TimePoint")

        #print(f"Saved all amplitudes for all replicates to {output_path}")
        return processed_data_IT

    @staticmethod
    def save_IT_profile_plot(spheroid_file, output_path):
        """
        Save a plot of the IT profile to the output directory.

        Args:
            spheroid_file: File with visualizable IT data.
            output_path (str): Path to store the plot.

        Returns:
            None
        """
        output_folder = os.path.join(output_path, "plots")
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, "save_IT_profile_plot.png")
        spheroid_file.visualize_IT_profile(save_path=save_path)
        
    
    @staticmethod
    def save_color_plot(spheroid_file, output_path):
        """
        Save a color-coded data visualization for the given file.

        Args:
            spheroid_file: File containing the color plot data.
            output_path (str): Path to output the color plot PNG.

        Returns:
            None
        """
        output_folder = os.path.join(output_path, "plots")
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, "color_plot.png")
        # Honor the manual upper colorbar limit set in the UI, if any, so bulk
        # exports match the on-screen / per-file exports.
        from PyQt5.QtCore import QSettings
        qs = QSettings("HashemiLab", "NeuroStemVolt")
        vmax = None
        if qs.value("color_vmax_manual", False, type=bool):
            try:
                vmax = float(qs.value("color_vmax")) or None
            except (TypeError, ValueError):
                vmax = None
        spheroid_file.visualize_color_plot_data(save_path=save_path, vmax=vmax)

    ### Methods for group_analysis
    @staticmethod
    def save_mean_ITs_plot(group_analysis, output_path):
        """
        Save a plot of mean ITs over time for all replicates.

        Args:
            group_analysis: GroupAnalysis instance with experiments.
            output_path (str): Directory to store the plot.

        Returns:
            None
        """
        output_folder = os.path.join(output_path, "plots")
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, "mean_ITs.png")
        group_analysis.plot_mean_ITs(save_path=save_path)

    @staticmethod
    def save_unprocessed_first_ITs_plot(group_analysis, output_path):
        """
        Save a plot showing the raw (unprocessed) ITs from the first stimulation.

        Args:
            group_analysis: The GroupAnalysis instance.
            output_path (str): Path to save the plot.

        Returns:
            None
        """
        output_folder = os.path.join(output_path, "plots")
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, "unprocessed_first_ITs_plot.png")
        group_analysis.plot_unprocessed_first_ITs(save_path=save_path)

    @staticmethod
    def save_plot_tau_over_time(group_analysis, output_path):
        """
        Save the plot of decay constant (tau) over all replicate time points.

        Args:
            group_analysis: GroupAnalysis object.
            output_path (str): Output directory for the plot.

        Returns:
            None
        """
        output_folder = os.path.join(output_path, "plots")
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, "plot_tau_over_time.png")
        group_analysis.plot_tau_over_time(save_path=save_path)

    @staticmethod
    def save_plot_exponential_fit_aligned(group_analysis, output_path, replicated_time_point=0):
        """
        Save exponential decay fit plot for a specific time point across replicates.

        Args:
            group_analysis: GroupAnalysis object.
            output_path (str): Directory to save the figure.
            replicated_time_point (int): Index of the replicate file to analyze.

        Returns:
            None
        """
        output_folder = os.path.join(output_path, "plots")
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, "plot_exponential_fit.png")
        group_analysis.plot_exponential_fit_aligned(save_path=save_path, replicate_time_point=replicated_time_point)

    #@staticmethod
    #def save_plot_amplitudes_over_time_single_experiment(group_analysis, output_path):
        #output_folder = os.path.join(output_path, "plots")
        #os.makedirs(output_folder, exist_ok=True)
        #group_analysis.save_plot_amplitudes_over_time_single_experiment(save_path=output_folder)
    
    @staticmethod
    def save_plot_all_amplitudes_over_time(group_analysis, output_path):
        """
        Save a line plot of amplitude evolution for each experiment.

        Args:
            group_analysis: GroupAnalysis instance.
            output_path (str): Output folder path.

        Returns:
            None
        """
        output_folder = os.path.join(output_path, "plots")
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, "plot_all_amplitudes_over_time.png")
        group_analysis.plot_all_amplitudes_over_time(save_path=save_path)

    @staticmethod
    def save_plot_mean_amplitudes_over_time(group_analysis, output_path):
        """
        Save a plot of mean amplitude over time with standard deviation shaded.

        Args:
            group_analysis: GroupAnalysis object.
            output_path (str): Directory to save the PNG.

        Returns:
            None
        """
        output_folder = os.path.join(output_path, "plots")
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, "plot_mean_amplitudes_over_time.png")
        group_analysis.plot_mean_amplitudes_over_time(save_path=save_path)

    @staticmethod
    def save_plot_first_stim_amplitudes(group_analysis, output_path):
        """
        Save a bar plot of unnormalized first-stim amplitudes across replicates.

        Args:
            group_analysis: GroupAnalysis containing replicate data.
            output_path (str): Directory to store the plot.

        Returns:
            None
        """
        output_folder = os.path.join(output_path, "plots")
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, "plot_first_stim_amplitudes.png")
        group_analysis.plot_first_stim_amplitudes(save_path=save_path)

    @staticmethod
    def save_experiment_log(group_experiments: GroupAnalysis, output_folder_path, qsettings=None):
        """
        Generate and save a comprehensive log file documenting all experiment metadata,
        settings, processing steps, and data provenance.

        Args:
            group_experiments (GroupAnalysis): Group containing all experiment replicates.
            output_folder_path (str): Directory where the log will be saved.
            qsettings (QSettings, optional): Qt settings object containing user configuration.

        Returns:
            str: Path to the saved log file, or None if no experiments exist.
        """
        from PyQt5.QtCore import QSettings

        experiments = group_experiments.get_experiments()
        if not experiments:
            return None

        if qsettings is None:
            qsettings = QSettings("HashemiLab", "NeuroStemVolt")

        log_folder = os.path.join(output_folder_path, "experiment_logs")
        os.makedirs(log_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_folder, f"experiment_log_{timestamp}.txt")

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NEUROSTEMVOLT EXPERIMENT LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Folder: {output_folder_path}\n")
            f.write("=" * 80 + "\n\n")

            # ===== EXPERIMENT CONFIGURATION =====
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("=" * 80 + "\n\n")

            file_type = qsettings.value("file_type", "Unknown", type=str)
            acquisition_freq = qsettings.value("acquisition_frequency", "Not set", type=str)
            file_length = qsettings.value("file_length", "Not set", type=str)
            peak_position = qsettings.value("peak_position", "Not set", type=str)
            waveform = qsettings.value("waveform", "Not set", type=str)
            treatment = qsettings.value("treatment", "Not set", type=str)
            time_between_files = qsettings.value("time_between_files", "Not set", type=str)
            files_before_treatment = qsettings.value("files_before_treatment", "Not set", type=str)

            f.write(f"File Type: {file_type}\n")
            f.write(f"Waveform: {waveform}\n")
            f.write(f"Treatment: {treatment}\n")
            f.write(f"Acquisition Frequency: {acquisition_freq} Hz\n")
            f.write(f"File Length: {file_length} seconds\n")
            f.write(f"Peak Position (voltage index): {peak_position}\n")
            f.write(f"Time Between Files: {time_between_files} minutes\n")
            f.write(f"Files Before Treatment: {files_before_treatment}\n\n")

            # Stimulation parameters
            if file_type != "Spontaneous":
                f.write("-" * 80 + "\n")
                f.write("STIMULATION PARAMETERS\n")
                f.write("-" * 80 + "\n")
                try:
                    stim_params_str = qsettings.value("stim_params", "{}")
                    stim_params = json.loads(stim_params_str) if stim_params_str else {}
                    if stim_params:
                        f.write(f"Stimulation Start: {stim_params.get('start', 'N/A')} seconds\n")
                        f.write(f"Stimulation Duration: {stim_params.get('duration', 'N/A')} seconds\n")
                        f.write(f"Stimulation Frequency: {stim_params.get('frequency', 'N/A')} Hz\n")
                        f.write(f"Stimulation Amplitude: {stim_params.get('amplitude', 'N/A')} V\n")
                        f.write(f"Number of Pulses: {stim_params.get('pulses', 'N/A')}\n")
                    else:
                        f.write("No stimulation parameters configured.\n")
                except Exception as e:
                    f.write(f"Error reading stimulation parameters: {e}\n")
                f.write("\n")

            # Calibration settings
            f.write("-" * 80 + "\n")
            f.write("CALIBRATION SETTINGS\n")
            f.write("-" * 80 + "\n")
            calibration_enabled = qsettings.value("calibration_enabled", False, type=bool)
            unit = "nM" if calibration_enabled else "nA"
            f.write(f"Calibration Enabled: {calibration_enabled}\n")
            if calibration_enabled:
                slope = qsettings.value("calibration_slope", 1.0, type=float)
                intercept = qsettings.value("calibration_intercept", 0.0, type=float)
                f.write(f"Slope: {slope}\n")
                f.write(f"Y-intercept: {intercept}\n")
                f.write(f"Conversion Formula: Concentration = (Current - {intercept}) / {slope}\n")
                f.write(f"Data is in concentration units ({unit}).\n")
            else:
                f.write(f"Data is in raw current units ({unit}).\n")
            f.write("\n")

            # ===== PROCESSING PIPELINE =====
            f.write("=" * 80 + "\n")
            f.write("PROCESSING PIPELINE\n")
            f.write("=" * 80 + "\n\n")

            try:
                pipeline_str = qsettings.value("processing_pipeline", "[]")
                pipeline = json.loads(pipeline_str) if pipeline_str else []
                params_str = qsettings.value("processing_params", "{}")
                params = json.loads(params_str) if params_str else {}

                if pipeline:
                    f.write("Applied Processing Steps (in order):\n\n")
                    for idx, step in enumerate(pipeline, 1):
                        f.write(f"{idx}. {step}\n")
                        if step in params:
                            param_info = params[step]
                            if isinstance(param_info, dict):
                                if step == "Artifact Removal":
                                    f.write(f"   - Threshold (MAD multiplier): {param_info.get('threshold', 'N/A')}\n")
                                    f.write(f"   - Pad (extra scans per edge): {param_info.get('pad', 'N/A')}\n")
                                    max_scans = param_info.get('max_artifact_scans', '0')
                                    max_scans_label = "auto (2 × acquisition frequency)" if str(max_scans) == "0" else max_scans
                                    f.write(f"   - Max Artifact Scans: {max_scans_label}\n")
                                elif step == "Multiple Peak Detection":
                                    f.write(f"   - Max Peaks: {param_info.get('max_peaks', 'N/A')}\n")
                                    f.write(f"   - Min Prominence: {param_info.get('min_prominence', 'N/A')}\n")
                                    f.write(f"   - CV Peak: {param_info.get('cv_peak', 'N/A')}\n")
                                    f.write(f"   - Peak Height Threshold: {param_info.get('peak_height_threshold', 'N/A')}\n")
                                else:
                                    for key, value in param_info.items():
                                        f.write(f"   - {key}: {value}\n")
                            elif isinstance(param_info, (list, tuple)):
                                if step == "Background Subtraction":
                                    f.write(f"   - Region: start={param_info[0]}s, end={param_info[1]}s\n")
                                elif step == "Savitzky-Golay Filter":
                                    f.write(f"   - Window: {param_info[0]}, Order: {param_info[1]}\n")
                                elif step == "Butterworth Filter":
                                    f.write(f"   - Order (p): {param_info[0]}\n")
                                    f.write(f"   - Cutoff cx: {param_info[1]} Hz\n")
                                    f.write(f"   - Cutoff cy: {param_info[2]} Hz\n")
                                else:
                                    f.write(f"   - Parameters: {', '.join(str(p) for p in param_info)}\n")
                            else:
                                f.write(f"   - Parameter: {param_info}\n")
                        f.write("\n")
                else:
                    f.write("No processing steps configured.\n")
            except Exception as e:
                f.write(f"Error reading processing pipeline: {e}\n")
            f.write("\n")

            # ===== SUMMARY =====
            f.write("=" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            try:
                total_files = sum(exp.get_file_count() for exp in experiments)
                f.write(f"Total Files Processed: {total_files}\n")
                folders = set()
                for exp in experiments:
                    for file_idx in range(exp.get_file_count()):
                        try:
                            filepath = exp.get_spheroid_file(file_idx).get_filepath()
                            folders.add(os.path.dirname(filepath))
                        except Exception:
                            pass
                f.write(f"Source Folders: {len(folders)}\n")
                for folder in sorted(folders):
                    f.write(f"  - {folder}\n")
            except Exception as e:
                f.write(f"Error computing summary: {e}\n")
            f.write("\n")

            # ===== REPLICATE DATA =====
            f.write("=" * 80 + "\n")
            f.write("REPLICATE DATA\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Number of Replicates: {len(experiments)}\n\n")

            for exp_idx, exp in enumerate(experiments, 1):
                f.write("-" * 80 + "\n")
                f.write(f"REPLICATE {exp_idx}\n")
                f.write("-" * 80 + "\n")
                try:
                    f.write(f"Treatment: {getattr(exp, 'treatment', 'N/A')}\n")
                    f.write(f"Waveform: {getattr(exp, 'waveform', 'N/A')}\n")
                    f.write(f"Number of Files (Timepoints): {exp.get_file_count()}\n")
                    f.write(f"File Length: {exp.get_file_length()} seconds\n")
                    f.write(f"Acquisition Frequency: {exp.get_acquisition_frequency()} Hz\n")
                    f.write(f"Time Between Files: {exp.get_time_between_files()} minutes\n")
                    f.write(f"Files Before Treatment: {exp.get_number_of_files_before_treatment()}\n")
                except Exception as e:
                    f.write(f"Error retrieving experiment info: {e}\n")
                f.write("\n")

                f.write("Data Files:\n")
                try:
                    for file_idx in range(exp.get_file_count()):
                        sf = exp.get_spheroid_file(file_idx)
                        filepath = sf.get_filepath()
                        filename = os.path.basename(filepath)
                        time_min = file_idx * exp.get_time_between_files()
                        baseline_marker = " [BASELINE]" if file_idx < exp.get_number_of_files_before_treatment() else ""
                        f.write(f"  {file_idx + 1}. {filename} (t={time_min} min){baseline_marker}\n")
                        f.write(f"     Path: {filepath}\n")
                        try:
                            meta = sf.get_metadata()
                            if meta:
                                # Manual peak edit flag
                                if meta.get('peak_manually_edited', False):
                                    f.write(f"     ** PEAK MANUALLY EDITED — results may differ from automated detection **\n")

                                # Peak amplitude
                                if 'peak_amplitude_values' in meta and meta['peak_amplitude_values'] is not None:
                                    amp_val = meta['peak_amplitude_values']
                                    if isinstance(amp_val, (list, np.ndarray)):
                                        if len(amp_val) > 0:
                                            if file_type == "Spontaneous":
                                                f.write(f"     Peaks Detected: {len(amp_val)}\n")
                                                f.write(f"     Mean Amplitude: {np.mean(amp_val):.4f} {unit}\n")
                                            else:
                                                f.write(f"     Peak Amplitude: {amp_val[0]:.4f} {unit}\n")
                                    else:
                                        f.write(f"     Peak Amplitude: {amp_val:.4f} {unit}\n")

                                # Peak position
                                if 'peak_amplitude_positions' in meta and meta['peak_amplitude_positions'] is not None:
                                    pos_val = meta['peak_amplitude_positions']
                                    acq_freq = exp.get_acquisition_frequency() or 1
                                    if isinstance(pos_val, (list, np.ndarray)) and len(pos_val) > 0:
                                        if file_type != "Spontaneous":
                                            f.write(f"     Peak Position: {pos_val[0]} samples ({pos_val[0] / acq_freq:.2f} s)\n")
                                    elif not isinstance(pos_val, (list, np.ndarray)):
                                        f.write(f"     Peak Position: {pos_val} samples ({pos_val / acq_freq:.2f} s)\n")

                                # Exponential fitting
                                if file_type != "Spontaneous" and 'exponential fitting parameters' in meta:
                                    fit_params = meta['exponential fitting parameters']
                                    if fit_params and isinstance(fit_params, dict):
                                        f.write(f"     Exponential Fit:\n")
                                        f.write(f"       - Amplitude (A): {fit_params.get('A', 'N/A'):.4f}\n")
                                        f.write(f"       - Tau (τ): {fit_params.get('tau', 'N/A'):.4f}\n")
                                        f.write(f"       - Constant (C): {fit_params.get('C', 'N/A'):.4f}\n")
                                        f.write(f"       - Half-life (t½): {fit_params.get('t_half', 'N/A'):.4f}\n")

                                # Baseline
                                if 'baseline' in meta and meta['baseline'] is not None:
                                    baseline = meta['baseline']
                                    if isinstance(baseline, (list, np.ndarray)) and len(baseline) > 0:
                                        f.write(f"     Baseline: {baseline[0]:.4f} {unit}\n")
                                    elif not isinstance(baseline, (list, np.ndarray)):
                                        f.write(f"     Baseline: {baseline:.4f} {unit}\n")
                        except Exception as e:
                            f.write(f"     Error reading metadata: {e}\n")
                        f.write("\n")
                except Exception as e:
                    f.write(f"Error listing files: {e}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF LOG\n")
            f.write("=" * 80 + "\n")

        print(f"Experiment log saved to: {log_path}")
        return log_path


if __name__ == "__main__":
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

    group_analysis = GroupAnalysis()
    group_analysis.add_experiment(experiment_one,experiment_two)

    # Save ITs
    output_folder = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/output"
    OutputManager.save_all_ITs(group_analysis,output_folder)
    OutputManager.save_all_peak_amplitudes(group_analysis,output_folder)
    #OutputManager.save_original_ITs(group_analysis,output_folder)
    #OutputManager.save_peak_amplitudes_metrics(group_analysis,output_folder)
    OutputManager.save_all_reuptake_curves(group_analysis,output_folder)
    OutputManager.save_all_exponential_fitting_params(group_analysis,output_folder)

    # 2. Save group-level plots
    OutputManager.save_mean_ITs_plot(group_analysis, output_folder)
    OutputManager.save_unprocessed_first_ITs_plot(group_analysis, output_folder)
    OutputManager.save_plot_tau_over_time(group_analysis, output_folder)
    #OutputManager.save_plot_amplitudes_over_time_single_experiment(group_analysis, output_folder)
    OutputManager.save_plot_mean_amplitudes_over_time(group_analysis, output_folder)
    OutputManager.save_plot_all_amplitudes_over_time(group_analysis, output_folder)
    OutputManager.save_plot_exponential_fit_aligned(group_analysis, output_folder)
    #OutputManager.save_plot_first_stim_amplitudes(group_analysis, output_folder)


