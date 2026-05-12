from core.spheroid_experiment import SpheroidExperiment
from core.group_analysis import GroupAnalysis
import os
import pandas as pd
import numpy as np
from scipy.stats import sem

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
            output_csv = "All_ITs_experiment_n{0}.csv".format(i)
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
        output_path = os.path.join(output_IT_folder, "All_ITs_all_replicates.csv")
        df.to_csv(output_path)
        print(f"Saved all ITs for all replicates to {output_path}")

        df_paired = df.swaplevel("Replicate","File", axis=1)   \
                      .sort_index(axis=1)                    

        paired_folder = os.path.join(output_folder_path,
                                     "all_replicates_ITs_paired")
        os.makedirs(paired_folder, exist_ok=True)
        paired_path = os.path.join(paired_folder,
                                   "All_ITs_all_replicates_paired_by_file.csv")
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
        output_path = os.path.join(output_folder, "All_amplitudes_all_replicates.csv")
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


