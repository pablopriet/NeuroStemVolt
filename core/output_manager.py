from core.spheroid_experiment import SpheroidExperiment
from core.group_analysis import GroupAnalysis
import os
import pandas as pd
import numpy as np
from scipy.stats import sem
from datetime import datetime
import json

class OutputManager:
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

        for i, experiment in enumerate(group_experiments.get_experiments()):
            it_matrix = []
            file_names = []
            # collect ITs and file names
            for j, spheroid_file in enumerate(experiment.files):
                IT_individual = spheroid_file.get_processed_data_IT()
                if IT_individual is None:
                    IT_individual = np.array([], dtype=float)
                it_matrix.append(np.asarray(IT_individual, dtype=float))
                file_name = spheroid_file.get_filepath()
                file_names.append(file_name)
            # determine max length and pad with zeros
            if len(it_matrix) == 0:
                continue
            max_len = max(arr.size for arr in it_matrix)
            try:
                it_matrix_padded = [np.pad(arr, (0, max_len - arr.size), mode='edge') for arr in it_matrix]
            except Exception:
                it_matrix_padded = [np.pad(arr, (0, max_len - arr.size), mode='constant', constant_values=0.0) for arr in it_matrix]
            # Transpose so each column is a file
            df = pd.DataFrame(it_matrix_padded).T
            df.columns = [os.path.basename(f) for f in file_names]
            output_csv = "All_ITs_experiment_n{0}.csv".format(i)
            output_IT_folder = os.path.join(output_folder_path,"replicate_ITs")
            os.makedirs(output_IT_folder, exist_ok=True)
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
        if not experiments:
            return None

        # Build list of columns (replicate, file) and collect all IT arrays
        arrays = []
        it_series = []
        for exp_idx, experiment in enumerate(experiments):
            rep_name = f"Rep{exp_idx+1}"
            for file_idx, spheroid_file in enumerate(experiment.files):
                file_short = os.path.basename(spheroid_file.get_filepath())
                arrays.append((rep_name, file_short))
                IT_individual = spheroid_file.get_processed_data_IT()
                if IT_individual is None:
                    IT_individual = np.array([], dtype=float)
                it_series.append(np.asarray(IT_individual, dtype=float))

        if len(it_series) == 0:
            return None

        # Determine max length across all ITs and pad with zeros
        max_len = max(arr.size for arr in it_series)
        try:
            padded_series = [np.pad(arr, (0, max_len - arr.size), mode='edge') for arr in it_series]
        except Exception:
            padded_series = [np.pad(arr, (0, max_len - arr.size), mode='constant', constant_values=0.0) for arr in it_series]
        data = np.column_stack(padded_series)

        # Create time axis in seconds (use acquisition freq from first experiment if available)
        try:
            acq_freq = experiments[0].get_acquisition_frequency()
            time_seconds = np.arange(max_len) / acq_freq
        except Exception:
            time_seconds = np.arange(max_len)

        # Create MultiIndex columns and DataFrame
        columns = pd.MultiIndex.from_tuples(arrays, names=["Replicate", "File"])
        df = pd.DataFrame(data, columns=columns)
        df.index = time_seconds
        df.index.name = "Time (s)"

        # Save to CSV
        output_IT_folder = os.path.join(output_folder_path, "all_replicates_ITs")
        os.makedirs(output_IT_folder, exist_ok=True)
        output_path = os.path.join(output_IT_folder, "All_ITs_all_replicates.csv")
        df.to_csv(output_path)
        print(f"Saved all ITs for all replicates to {output_path}")

        # paired view
        try:
            df_paired = df.swaplevel("Replicate","File", axis=1).sort_index(axis=1)
            paired_folder = os.path.join(output_folder_path, "all_replicates_ITs_paired")
            os.makedirs(paired_folder, exist_ok=True)
            paired_path = os.path.join(paired_folder, "All_ITs_all_replicates_paired_by_file.csv")
            df_paired.to_csv(paired_path)
            print(f"Saved paired ITs (same file side-by-side) to {paired_path}")
        except Exception:
            pass
        
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
            output_csv = "Original_ITs_experiment_n{0}.csv".format(i)
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
                output_csv = "Files_Amplitudes_experiment_n{0}.csv".format(i)
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
                output_detailed = f"Experiment_{i+1}_Detailed_Peak_Data.csv"
                df_detailed.to_csv(os.path.join(spont_folder, output_detailed), index=False)

            # Save summary for this experiment
            if summary_records:
                df_summary = pd.DataFrame(summary_records)
                output_summary = f"Experiment_{i+1}_Summary_Peak_Data.csv"
                df_summary.to_csv(os.path.join(spont_folder, output_summary), index=False)

                # Add to all experiments summary
                for record in summary_records:
                    record['Experiment'] = i + 1
                    all_summary_records.append(record)

        # Save combined summary for all experiments
        if all_summary_records:
            df_all = pd.DataFrame(all_summary_records)
            output_all = "All_Experiments_Spontaneous_Peak_Summary.csv"
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
                output_timepoints = "Group_Statistics_By_Timepoint.csv"
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
        if not experiments:
            print("No experiments available to export peak amplitudes.")
            return None

        n_experiments = len(experiments)
        # determine maximum number of files across experiments
        max_files = max(getattr(exp, 'get_file_count', lambda: len(getattr(exp, 'files', [])))() for exp in experiments)

        # timing parameters (use first experiment defaults if present)
        try:
            n_before = experiments[0].get_number_of_files_before_treatment()
            interval = experiments[0].get_time_between_files()
        except Exception:
            n_before = 0
            interval = 0

        # build time axis using max_files
        if n_before > 0:
            time_points = [interval * (i - n_before) for i in range(max_files)]
        else:
            time_points = [i * interval for i in range(max_files)]

        # prepare DataFrames indexed by file index, columns per replicate
        rep_cols = [f"Rep{idx+1}" for idx in range(n_experiments)]
        df_amp = pd.DataFrame(index=range(max_files), columns=rep_cols, dtype=float)
        df_pos = pd.DataFrame(index=range(max_files), columns=rep_cols, dtype=float)

        # fill with NaN by default (already NaN)
        for exp_idx, exp in enumerate(experiments):
            col = f"Rep{exp_idx+1}"
            try:
                file_count = exp.get_file_count()
            except Exception:
                file_count = len(getattr(exp, "files", []))
            for file_idx in range(file_count):
                try:
                    sf = exp.get_spheroid_file(file_idx)
                except Exception:
                    continue
                try:
                    meta = sf.get_metadata() or {}
                except Exception:
                    meta = {}

                # amplitude
                amp_raw = meta.get('peak_amplitude_values', None)
                amp_val = None
                if amp_raw is not None:
                    try:
                        arr = np.asarray(amp_raw)
                        if arr.size == 0:
                            amp_val = np.nan
                        else:
                            amp_val = float(arr.ravel()[0])
                    except Exception:
                        try:
                            amp_val = float(amp_raw)
                        except Exception:
                            amp_val = np.nan
                # position (convert to seconds if numeric)
                pos_raw = meta.get('peak_amplitude_positions', None)
                pos_val = None
                if pos_raw is not None:
                    try:
                        parr = np.asarray(pos_raw)
                        if parr.size == 0:
                            pos_val = np.nan
                        else:
                            pos_val = float(parr.ravel()[0]) / (experiments[0].get_acquisition_frequency() or 1.0)
                    except Exception:
                        try:
                            pos_val = float(pos_raw) / (experiments[0].get_acquisition_frequency() or 1.0)
                        except Exception:
                            pos_val = np.nan

                # assign into DataFrames (leave NaN if missing)
                if amp_val is not None:
                    df_amp.at[file_idx, col] = amp_val
                if pos_val is not None:
                    df_pos.at[file_idx, col] = pos_val

        # combine amplitude and position into a single multi-column DataFrame
        combined = pd.concat({'Amplitude': df_amp, 'Position (s)': df_pos}, axis=1)
        # insert Time column
        combined.insert(0, "Time", time_points[:combined.shape[0]])

        # save
        output_folder = os.path.join(output_folder_path, "all_replicates_amplitudes")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "All_amplitudes_all_replicates.csv")
        combined.to_csv(output_path, index=False)
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
        output_path = os.path.join(output_folder, "All_AUC_all_replicates.csv")
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
        mean_output_path = os.path.join(output_folder, "Mean_AUC_with_SEM.csv")
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
        output_path = os.path.join(output_IT_folder, "All_reuptakes.csv")
        df.to_csv(output_path)
        print(f"Saved all reuptakes for all replicates to {output_path}")

        df_paired = df.swaplevel("Replicate","File", axis=1)   \
                      .sort_index(axis=1)                    

        paired_folder = os.path.join(output_folder_path,
                                     "all_reuptakes_paired")
        os.makedirs(paired_folder, exist_ok=True)
        paired_path = os.path.join(paired_folder,
                                   "All_reuptakes_paired_by_file.csv")
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

        # Save to CSV
        output_folder = os.path.join(output_folder_path, "all_exponential_fit_params")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "all_exp_fit_params.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved all params for all replicates to {output_path}")

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
        output_file_name = os.path.join(base_name + "_IT.csv")
        
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
    def save_color_plot_mean_data(group_experiments : GroupAnalysis, output_folder_path):
        experiments = group_analysis.get_experiments()
        for exp_idx, experiment in enumerate(experiments):
            if not experiments:
                return None

        file_count = experiments[0].get_file_count()
        acq_freq = experiments[0].get_acquisition_frequency()
        file_length = experiments[0].get_file_length()
        n_timepoints = experiments[0].get_file_time_points()

        # Get column headers from first experiment's file base names
        base_names = [os.path.splitext(os.path.basename(sf.get_filepath()))[0] for sf in experiments[0].files]

        # Collect ITs for each file index across experiments
        mean_ITs = []
        for file_idx in range(file_count):
            # Gather ITs for this file index from all experiments
            it_matrix = []
            for exp in experiments:
                IT_individual = exp.get_spheroid_file(file_idx).get_processed_data()
                it_matrix.append(IT_individual)
            # Pad to same length if needed
            max_len = max(len(it) for it in it_matrix)
            it_matrix_padded = [np.pad(it, (0, max_len - len(it)), constant_values=np.nan) for it in it_matrix]
            # Average across experiments (axis=0)
            mean_IT = np.nanmean(it_matrix_padded, axis=0)
            mean_ITs.append(mean_IT)

        # Transpose so each column is a file index, each row is a timepoint
        mean_ITs_array = np.array(mean_ITs).T  # shape: (n_timepoints, file_count)

        # Create time axis in seconds
        time_seconds = np.arange(mean_ITs_array.shape[0]) / acq_freq

        # Build DataFrame
        df = pd.DataFrame(mean_ITs_array, columns=base_names)
        df.insert(0, "Time (s)", time_seconds)

        # Save to CSV
        output_folder = os.path.join(output_folder_path, "mean_ITs")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "Mean_ITs_across_experiments.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved mean ITs across experiments to {output_path}")

    
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
        spheroid_file.visualize_color_plot_data(save_path=save_path)

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
    def save_plot_frequency_over_time(group_analysis, output_path):
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
        save_path = os.path.join(output_folder, "plot_frequency_over_time.png")
        group_analysis.plot_frequency_over_time(save_path=save_path)

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
    def save_mean_ITs(group_experiments: GroupAnalysis, output_folder_path):
        """
        Save the mean IT trace for each file index across all experiments into a single CSV file.
        Each column corresponds to a file index (e.g., timepoint), using the base filename from the first experiment.

        Args:
            group_experiments (GroupAnalysis): Group containing multiple experiments.
            output_folder_path (str): Directory to save the output file.

        Returns:
            None
        """
        experiments = group_experiments.get_experiments()
        if not experiments:
            return None

        file_count = experiments[0].get_file_count()
        acq_freq = experiments[0].get_acquisition_frequency()
        file_length = experiments[0].get_file_length()
        n_timepoints = experiments[0].get_file_time_points()

        # Get column headers from first experiment's file base names
        base_names = [os.path.splitext(os.path.basename(sf.get_filepath()))[0] for sf in experiments[0].files]

        # Collect ITs for each file index across experiments
        mean_ITs = []
        for file_idx in range(file_count):
            # Gather ITs for this file index from all experiments
            it_matrix = []
            for exp in experiments:
                IT_individual = exp.get_spheroid_file(file_idx).get_processed_data_IT()
                it_matrix.append(IT_individual)
            # Pad to same length if needed
            max_len = max(len(it) for it in it_matrix)
            try:
                it_matrix_padded = [np.pad(it, (0, max_len - len(it)), mode='edge') for it in it_matrix]
            except Exception:
                it_matrix_padded = [np.pad(it, (0, max_len - len(it)), mode='constant', constant_values=np.nan) for it in it_matrix]
            # Average across experiments (axis=0)
            mean_IT = np.nanmean(it_matrix_padded, axis=0)
            mean_ITs.append(mean_IT)

        # Transpose so each column is a file index, each row is a timepoint
        mean_ITs_array = np.array(mean_ITs).T  # shape: (n_timepoints, file_count)

        # Create time axis in seconds
        time_seconds = np.arange(mean_ITs_array.shape[0]) / acq_freq

        # Build DataFrame
        df = pd.DataFrame(mean_ITs_array, columns=base_names)
        df.insert(0, "Time (s)", time_seconds)

        # Save to CSV
        output_folder = os.path.join(output_folder_path, "mean_ITs")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "Mean_ITs_across_experiments.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved mean ITs across experiments to {output_path}")

    @staticmethod
    def save_mean_processed_data_matrices(group_experiments: GroupAnalysis, output_folder_path):
        """
        For each file index, compute the mean processed data matrix across all experiments,
        and save each as an individual CSV named after the base file name (from the first experiment).

        Args:
            group_experiments (GroupAnalysis): Group containing multiple experiments.
            output_folder_path (str): Directory to save the output files.

        Returns:
            None
        """
        experiments = group_experiments.get_experiments()
        if not experiments:
            return None

        file_count = experiments[0].get_file_count()
        base_names = [os.path.splitext(os.path.basename(sf.get_filepath()))[0] for sf in experiments[0].files]

        output_folder = os.path.join(output_folder_path, "mean_processed_matrices")
        os.makedirs(output_folder, exist_ok=True)

        for file_idx in range(file_count):
            # Gather processed data matrices for this file index from all experiments
            matrices = []
            for exp in experiments:
                matrix = exp.get_spheroid_file(file_idx).get_processed_data()
                matrices.append(matrix)
            # Pad matrices to the same shape if needed
            shapes = [m.shape for m in matrices]
            max_shape = (max(s[0] for s in shapes), max(s[1] for s in shapes))
            try:
                matrices_padded = [
                    np.pad(m, ((0, max_shape[0] - m.shape[0]), (0, max_shape[1] - m.shape[1])), mode='edge')
                    for m in matrices
                ]
            except Exception:
                matrices_padded = [
                    np.pad(m, ((0, max_shape[0] - m.shape[0]), (0, max_shape[1] - m.shape[1])), mode='constant', constant_values=np.nan)
                    for m in matrices
                ]
            # Compute mean matrix
            mean_matrix = np.nanmean(matrices_padded, axis=0).T
            # Save to CSV
            df = pd.DataFrame(mean_matrix)
            output_path = os.path.join(output_folder, f"{base_names[file_idx]}_mean_processed_matrix.csv")
            df.to_csv(output_path, index=False, header=False)
            print(f"Saved mean processed matrix for {base_names[file_idx]} to {output_path}")

    @staticmethod
    def save_experiment_log(group_experiments: GroupAnalysis, output_folder_path, qsettings=None):
        """
        Generate and save a comprehensive log file documenting all experiment metadata,
        settings, processing steps, and data provenance.

        This log captures:
        - Experiment configuration (acquisition parameters, waveform, treatment)
        - Stimulation parameters (if applicable)
        - Calibration settings
        - All data file paths and replicate organization
        - Processing pipeline steps with parameters
        - File-level metadata (peaks, amplitudes, exponential fit parameters)
        - Export timestamp and software version

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
            print("No experiments available to create log.")
            return None

        # Use provided qsettings or create new instance
        if qsettings is None:
            qsettings = QSettings("HashemiLab", "NeuroStemVolt")

        # Create log folder
        log_folder = os.path.join(output_folder_path, "experiment_logs")
        os.makedirs(log_folder, exist_ok=True)

        # Generate timestamp for log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"experiment_log_{timestamp}.txt"
        log_path = os.path.join(log_folder, log_filename)

        with open(log_path, 'w', encoding='utf-8') as log_file:
            # Header
            log_file.write("=" * 80 + "\n")
            log_file.write("NEUROSTEMVOLT EXPERIMENT LOG\n")
            log_file.write("=" * 80 + "\n")
            log_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Software Version: v1.0.0\n")
            log_file.write(f"Output Folder: {output_folder_path}\n")
            log_file.write("=" * 80 + "\n\n")

            # ===== EXPERIMENT CONFIGURATION =====
            log_file.write("=" * 80 + "\n")
            log_file.write("EXPERIMENT CONFIGURATION\n")
            log_file.write("=" * 80 + "\n\n")

            # Global settings
            file_type = qsettings.value("file_type", "Unknown", type=str)
            acquisition_freq = qsettings.value("acquisition_frequency", "Not set", type=str)
            file_length = qsettings.value("file_length", "Not set", type=str)
            peak_position = qsettings.value("peak_position", "Not set", type=str)
            waveform = qsettings.value("waveform", "Not set", type=str)
            treatment = qsettings.value("treatment", "Not set", type=str)
            time_between_files = qsettings.value("time_between_files", "Not set", type=str)
            files_before_treatment = qsettings.value("files_before_treatment", "Not set", type=str)

            log_file.write(f"File Type: {file_type}\n")
            log_file.write(f"Waveform: {waveform}\n")
            log_file.write(f"Treatment: {treatment}\n")
            log_file.write(f"Acquisition Frequency: {acquisition_freq} Hz\n")
            log_file.write(f"File Length: {file_length} seconds\n")
            log_file.write(f"Peak Position (voltage index): {peak_position}\n")
            log_file.write(f"Time Between Files: {time_between_files} minutes\n")
            log_file.write(f"Files Before Treatment: {files_before_treatment}\n\n")

            # Stimulation parameters (if applicable)
            if file_type != "Spontaneous":
                log_file.write("-" * 80 + "\n")
                log_file.write("STIMULATION PARAMETERS\n")
                log_file.write("-" * 80 + "\n")
                try:
                    stim_params_str = qsettings.value("stim_params", "{}")
                    stim_params = json.loads(stim_params_str) if stim_params_str else {}
                    
                    if stim_params:
                        log_file.write(f"Stimulation Start: {stim_params.get('start', 'N/A')} seconds\n")
                        log_file.write(f"Stimulation Duration: {stim_params.get('duration', 'N/A')} seconds\n")
                        log_file.write(f"Stimulation Frequency: {stim_params.get('frequency', 'N/A')} Hz\n")
                        log_file.write(f"Stimulation Amplitude: {stim_params.get('amplitude', 'N/A')} V\n")
                        log_file.write(f"Number of Pulses: {stim_params.get('pulses', 'N/A')}\n")
                    else:
                        log_file.write("No stimulation parameters configured.\n")
                except Exception as e:
                    log_file.write(f"Error reading stimulation parameters: {e}\n")
                log_file.write("\n")

            # Calibration settings
            log_file.write("-" * 80 + "\n")
            log_file.write("CALIBRATION SETTINGS\n")
            log_file.write("-" * 80 + "\n")
            calibration_enabled = qsettings.value("calibration_enabled", False, type=bool)
            log_file.write(f"Calibration Enabled: {calibration_enabled}\n")
            if calibration_enabled:
                slope = qsettings.value("calibration_slope", 1.0, type=float)
                intercept = qsettings.value("calibration_intercept", 0.0, type=float)
                log_file.write(f"Slope: {slope}\n")
                log_file.write(f"Y-intercept: {intercept}\n")
                log_file.write(f"Conversion Formula: Concentration = (Current - {intercept}) / {slope}\n")
            else:
                log_file.write("Data is in raw current units (nA).\n")
            log_file.write("\n")

            # ===== PROCESSING PIPELINE =====
            log_file.write("=" * 80 + "\n")
            log_file.write("PROCESSING PIPELINE\n")
            log_file.write("=" * 80 + "\n\n")

            try:
                pipeline_str = qsettings.value("processing_pipeline", "[]")
                pipeline = json.loads(pipeline_str) if pipeline_str else []
                
                params_str = qsettings.value("processing_params", "{}")
                params = json.loads(params_str) if params_str else {}

                if pipeline:
                    log_file.write("Applied Processing Steps (in order):\n\n")
                    for idx, step in enumerate(pipeline, 1):
                        log_file.write(f"{idx}. {step}\n")
                        if step in params:
                            param_info = params[step]
                            if isinstance(param_info, dict):
                                for key, value in param_info.items():
                                    log_file.write(f"   - {key}: {value}\n")
                            elif isinstance(param_info, (list, tuple)):
                                if step == "Background Subtraction":
                                    log_file.write(f"   - Region: start={param_info[0]}s, end={param_info[1]}s\n")
                                elif step == "Savitzky-Golay Filter":
                                    log_file.write(f"   - Window: {param_info[0]}, Order: {param_info[1]}\n")
                            else:
                                log_file.write(f"   - Parameter: {param_info}\n")
                        log_file.write("\n")
                else:
                    log_file.write("No processing steps configured or default pipeline used.\n")
            except Exception as e:
                log_file.write(f"Error reading processing pipeline: {e}\n")
            log_file.write("\n")

            # ===== REPLICATE DATA =====
            log_file.write("=" * 80 + "\n")
            log_file.write("REPLICATE DATA\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.write(f"Total Number of Replicates: {len(experiments)}\n\n")

            for exp_idx, exp in enumerate(experiments, 1):
                log_file.write("-" * 80 + "\n")
                log_file.write(f"REPLICATE {exp_idx}\n")
                log_file.write("-" * 80 + "\n")
                
                # Experiment-level info
                try:
                    log_file.write(f"Treatment: {getattr(exp, 'treatment', 'N/A')}\n")
                    log_file.write(f"Waveform: {getattr(exp, 'waveform', 'N/A')}\n")
                    log_file.write(f"Number of Files (Timepoints): {exp.get_file_count()}\n")
                    log_file.write(f"File Length: {exp.get_file_length()} seconds\n")
                    log_file.write(f"Acquisition Frequency: {exp.get_acquisition_frequency()} Hz\n")
                    log_file.write(f"Time Between Files: {exp.get_time_between_files()} minutes\n")
                    log_file.write(f"Files Before Treatment: {exp.get_number_of_files_before_treatment()}\n")
                except Exception as e:
                    log_file.write(f"Error retrieving experiment info: {e}\n")
                
                log_file.write("\n")

                # Data files
                log_file.write("Data Files:\n")
                try:
                    for file_idx in range(exp.get_file_count()):
                        sf = exp.get_spheroid_file(file_idx)
                        filepath = sf.get_filepath()
                        filename = os.path.basename(filepath)
                        
                        # Calculate time in minutes
                        time_min = file_idx * exp.get_time_between_files()
                        baseline_marker = " [BASELINE]" if file_idx < exp.get_number_of_files_before_treatment() else ""
                        
                        log_file.write(f"  {file_idx + 1}. {filename} (t={time_min} min){baseline_marker}\n")
                        log_file.write(f"     Path: {filepath}\n")
                        
                        # File metadata
                        try:
                            meta = sf.get_metadata()
                            if meta:
                                # Peak amplitude
                                if 'peak_amplitude_values' in meta and meta['peak_amplitude_values'] is not None:
                                    amp_val = meta['peak_amplitude_values']
                                    if isinstance(amp_val, (list, np.ndarray)):
                                        if len(amp_val) > 0:
                                            if file_type == "Spontaneous":
                                                log_file.write(f"     Peaks Detected: {len(amp_val)}\n")
                                                log_file.write(f"     Mean Amplitude: {np.mean(amp_val):.4f} nA\n")
                                            else:
                                                log_file.write(f"     Peak Amplitude: {amp_val[0]:.4f} nA\n")
                                    else:
                                        log_file.write(f"     Peak Amplitude: {amp_val:.4f} nA\n")
                                
                                # Peak position
                                if 'peak_amplitude_positions' in meta and meta['peak_amplitude_positions'] is not None:
                                    pos_val = meta['peak_amplitude_positions']
                                    if isinstance(pos_val, (list, np.ndarray)) and len(pos_val) > 0:
                                        if file_type != "Spontaneous":
                                            pos_sec = pos_val[0] / exp.get_acquisition_frequency()
                                            log_file.write(f"     Peak Position: {pos_val[0]} samples ({pos_sec:.2f} s)\n")
                                    elif not isinstance(pos_val, (list, np.ndarray)):
                                        pos_sec = pos_val / exp.get_acquisition_frequency()
                                        log_file.write(f"     Peak Position: {pos_val} samples ({pos_sec:.2f} s)\n")
                                
                                # Exponential fitting (for stimulated data)
                                if file_type != "Spontaneous" and 'exponential fitting parameters' in meta:
                                    fit_params = meta['exponential fitting parameters']
                                    if fit_params and isinstance(fit_params, dict):
                                        log_file.write(f"     Exponential Fit:\n")
                                        log_file.write(f"       - Amplitude (A): {fit_params.get('A', 'N/A'):.4f}\n")
                                        log_file.write(f"       - Tau (τ): {fit_params.get('tau', 'N/A'):.4f}\n")
                                        log_file.write(f"       - Constant (C): {fit_params.get('C', 'N/A'):.4f}\n")
                                        log_file.write(f"       - Half-life (t½): {fit_params.get('t_half', 'N/A'):.4f}\n")
                                
                                # Baseline info
                                if 'baseline' in meta and meta['baseline'] is not None:
                                    baseline = meta['baseline']
                                    if isinstance(baseline, (list, np.ndarray)) and len(baseline) > 0:
                                        log_file.write(f"     Baseline: {baseline[0]:.4f} nA\n")
                                    elif not isinstance(baseline, (list, np.ndarray)):
                                        log_file.write(f"     Baseline: {baseline:.4f} nA\n")
                        except Exception as e:
                            log_file.write(f"     Error reading metadata: {e}\n")
                        
                        log_file.write("\n")
                except Exception as e:
                    log_file.write(f"Error listing files: {e}\n")
                
                log_file.write("\n")

            # ===== SUMMARY =====
            log_file.write("=" * 80 + "\n")
            log_file.write("SUMMARY\n")
            log_file.write("=" * 80 + "\n\n")

            try:
                total_files = sum(exp.get_file_count() for exp in experiments)
                log_file.write(f"Total Files Processed: {total_files}\n")
                
                # Get unique folder paths
                folders = set()
                for exp in experiments:
                    for file_idx in range(exp.get_file_count()):
                        try:
                            filepath = exp.get_spheroid_file(file_idx).get_filepath()
                            folder = os.path.dirname(filepath)
                            folders.add(folder)
                        except:
                            pass
                
                log_file.write(f"Source Folders: {len(folders)}\n")
                for folder in sorted(folders):
                    log_file.write(f"  - {folder}\n")
                
            except Exception as e:
                log_file.write(f"Error computing summary statistics: {e}\n")

            log_file.write("\n")

            # Footer
            log_file.write("=" * 80 + "\n")
            log_file.write("END OF LOG\n")
            log_file.write("=" * 80 + "\n")

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


