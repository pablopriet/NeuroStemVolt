from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QProgressDialog, QApplication
from PyQt5.QtCore import QSettings
import numpy as np
from scipy.optimize import curve_fit

from PyQt5.QtCore import Qt

class PlotCanvas(FigureCanvas):
    """
    A Qt-compatible Matplotlib canvas for visualizing FSCV data.

    This class supports:
    - Color plot rendering
    - I-T profile visualization with peak annotation
    - Group-level amplitude and exponential decay visualization
    - Tau parameter tracking over time
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialize the PlotCanvas.

        Args:
            parent (QWidget, optional): Parent widget.
            width (float): Width of the figure in inches.
            height (float): Height of the figure in inches.
            dpi (int): Dots per inch for the figure.
        """
        import matplotlib

        matplotlib.rcParams.update({
            "font.family": ["Arial"],
            "font.size": 10,
        })
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig = fig
        self.axes = fig.add_subplot(111) 
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()
        self.cbar = None

    def plot_color(self, processed_data, peak_pos = None, title_suffix=None, vmax=None):
        """
        Display a 2D color plot of the FSCV data matrix.

        Args:
            processed_data (np.ndarray): 2D data array (voltage x time).
            peak_pos (int, optional): Index of the peak position for annotation.
            title_suffix (str, optional): Text to append to the plot title.
            vmax (float, optional): Manual upper color limit (nA). If None, auto-scales.
                The lower limit always stays automatic (-max|data|).

        Returns:
            None
        """
        from core.spheroid_file import PLOT_SETTINGS  # Ensure it's correctly imported

        plot_settings = PLOT_SETTINGS()
        custom_cmap = plot_settings.custom
        # Optional manual upper limit; the lower limit is fixed at -(2/3) of the top.
        norm = plot_settings.get_norm(processed_data, vmax=vmax)

        freq = QSettings("HashemiLab", "NeuroStemVolt").value("acquisition_frequency", 10, type=int)
        time_extent_seconds = processed_data.shape[0] / freq

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

        im = self.axes.imshow(
            processed_data.T,       # Transpose to align voltage steps as rows
            aspect='auto',
            cmap=custom_cmap,
            origin='lower',
            extent=[0, time_extent_seconds, 0, processed_data.shape[1]],
            norm=norm
        )
        if peak_pos != None:
            self.axes.axhline(
                y=peak_pos, 
                color='white', 
                linewidth=1, 
                linestyle='--'
            )
        # Add colorbar using the figure object (Qt-safe)
        self.cbar = self.fig.colorbar(im, ax=self.axes, label="Current (nA)")

        self.axes.set_xlabel("Time (seconds)")
        self.axes.set_ylabel("Voltage Steps")
        title = f"Color Plot{': ' + title_suffix if title_suffix else ''}"
        self.axes.set_title(title, fontweight='bold')       

        self.fig.tight_layout()
        self.draw()

    def plot_IT(self, processed_data, metadata=None, peak_position=None, temp_peak_detection=None,
               stim_start_sec=None, stim_duration_sec=None):
        """
        Plot the I-T (Current vs Time) trace, with optional peak annotations.

        Args:
            processed_data (np.ndarray): 2D data array (voltage x time).
            metadata (dict, optional): Dictionary with peak positions/values.
            peak_position (int, optional): Voltage index used for I-T extraction.

        Returns:
            None
        """
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)
        profile = processed_data[:, peak_position]
        n = profile.shape[0]
        t = np.arange(n) / freq  # in seconds

        # Plot the main profile
        self.axes.plot(t, profile, label="I-T Profile", color='#4178F2', linewidth=1.5)

        # Plot peak markers if metadata is provided
        if metadata and 'peak_amplitude_positions' in metadata:
            peak_indices = metadata['peak_amplitude_positions']
            peak_values = metadata.get('peak_amplitude_values', None)

            # Handle multiple peaks (list/array)
            if isinstance(peak_indices, (list, np.ndarray)) and len(peak_indices) > 0:
                # Multiple peaks detected
                if peak_values is None:
                    peak_values = [profile[idx] for idx in peak_indices if 0 <= idx < len(profile)]

                # Plot all peaks with different colors or numbering
                colors = ['#FF3877', '#FF8C00', '#32CD32', '#9370DB', '#FF69B4', '#00CED1', '#FFD700', '#DC143C', '#00FF7F',
                          '#FF1493']

                for i, (idx, val) in enumerate(zip(peak_indices, peak_values)):
                    if 0 <= idx < len(profile):
                        color = colors[i % len(colors)]  # Cycle through colors
                        self.axes.scatter(t[idx], val, color=color, s=100, zorder=5,
                                          label=f'Peak {i + 1}' if i < 5 else None)  # Limit legend entries

                        # Annotate with peak number and value
                        self.axes.annotate(f"P{i + 1}: {val:.2f}", (t[idx], val),
                                           textcoords="offset points", xytext=(0, 15),
                                           ha='center', fontsize=8, color=color, fontweight='bold')

                    # Add summary text
                    num_peaks = len(peak_indices)
                    self.axes.text(0.02, 0.98, f"Detected {num_peaks} peaks",
                                   transform=self.axes.transAxes, fontsize=10,
                                   verticalalignment='top', bbox=dict(boxstyle='round',
                                                                      facecolor='wheat', alpha=0.8))

            else:
                # Single peak (backward compatibility)
                try:
                    idx = int(peak_indices) if not isinstance(peak_indices, (list, np.ndarray)) else peak_indices[0]
                    val = float(peak_values) if not isinstance(peak_values, (list, np.ndarray)) else peak_values[0]
                    if 0 <= idx < len(profile):
                        self.axes.scatter(t[idx], val, color='#FF3877', s=100, alpha=0.5, zorder=5, label='Current Peak')
                        self.axes.annotate(f"{val:.2f}", (t[idx], val), textcoords="offset points",
                                           xytext=(0, 10), ha='center', fontsize=9, color='#FF3877')
                except (ValueError, TypeError, IndexError):
                    pass

            # Plot temporary peak detection position (preview from slider)
            if temp_peak_detection is not None and 0 <= temp_peak_detection < len(profile):
                temp_time = temp_peak_detection / freq
                temp_value = profile[temp_peak_detection]
                self.axes.scatter(temp_time, temp_value, color='#00FF00', s=120, zorder=6,
                                  label='Preview Peak', marker='o', alpha=0.5, linewidth=2)
                self.axes.annotate(f"Preview: {temp_value:.2f}", (temp_time, temp_value),
                                   textcoords="offset points", xytext=(0, -25),
                                   ha='center', fontsize=9, color='#00FF00', alpha=0.5, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#00FF00',
                                             alpha=0.8))

        # Add validation status to title
        validation_status = ""
        multiple_peaks_info = ""

        if metadata:
            # Check for multiple peaks
            if 'num_peaks_detected' in metadata:
                num_peaks = metadata['num_peaks_detected']
                if num_peaks > 1:
                    multiple_peaks_info = f" ({num_peaks} peaks)"

            # Check validation status
            if 'decay_validation_params' in metadata and metadata['decay_validation_params'] is not None:
                is_valid = metadata['decay_validation_params'].get('peak_passed_validation', False)
                validation_status = " (Valid)" if is_valid else " (Invalid)"

        # Axis labeling and formatting
        self.axes.set_xlabel("Time (seconds)")
        self.axes.set_ylabel("Current (nA)")
        title = f"I-T Profile{multiple_peaks_info}{validation_status}"
        if peak_position is not None:
            title += f" - Position {peak_position}"

        self.axes.set_title(title, fontweight="bold")
        self.axes.grid(True, alpha=0.3)

        # Stim reference overlay
        if stim_start_sec is not None:
            
            self.axes.axvline(x=stim_start_sec, color='#c0392b', linestyle='--',
                              linewidth=1.5, alpha=0.75, label=f'Stim start ({stim_start_sec:.1f}s)')
            if stim_duration_sec is not None and stim_duration_sec > 0:
                self.axes.axvspan(stim_start_sec, stim_start_sec + stim_duration_sec,
                                  alpha=0.15, color='#e67e22',
                                  label=f'Stim ({stim_duration_sec:.1f}s)')

        # Only show legend if there are items to show
        handles, labels = self.axes.get_legend_handles_labels()
        if handles:
            self.axes.legend(loc='upper right', fontsize=8)

        # Set x-axis ticks
        max_t = t[-1]
        tick_interval = 5  # seconds
        ticks = np.arange(0, max_t + tick_interval, tick_interval)
        self.axes.set_xticks(ticks)

        self.draw()

    def _plot_decay_regions(self, metadata, t, freq):
        """Plot decay regions on the current axes"""
        # Plot left decay region
        if 'decay_left_region' in metadata:
            left_data = metadata['decay_left_region']
            if 'indices' in left_data and 'values' in left_data:
                left_indices = list(left_data['indices'])
                left_values = left_data['values']
                if len(left_indices) > 0 and len(left_values) > 0:
                    left_times = [t[i] for i in left_indices if i < len(t)]
                    self.axes.plot(left_times, left_values, 'o-', color='orange', 
                                  alpha=0.8, markersize=4, label='Left Decay Region')

        # Plot right decay region
        if 'decay_right_region' in metadata:
            right_data = metadata['decay_right_region']
            if 'indices' in right_data and 'values' in right_data:
                right_indices = list(right_data['indices'])
                right_values = right_data['values']
                if len(right_indices) > 0 and len(right_values) > 0:
                    right_times = [t[i] for i in right_indices if i < len(t)]
                    self.axes.plot(right_times, right_values, 'o-', color='green', 
                                  alpha=0.8, markersize=4, label='Right Decay Region')

    # Keep all your existing methods unchanged
    def show_average_over_experiments(self, group_analysis):
        """
        Plot mean amplitude over time across all experiments with standard deviation.

        Args:
            group_analysis (GroupAnalysis): Backend object holding replicate experiments.

        Returns:
            None
        """
        import numpy as np

        # Show loading dialog
        progress = QProgressDialog("Processing data, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()  # Ensure dialog appears

        # Get data from group_analysis
        time_points, mean_amplitudes, all_amplitudes, files_before_treatment = group_analysis.amplitudes_over_time_all_experiments()
        
        if time_points is None:
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)
            self.axes.set_title("No data to plot")
            self.draw()
            progress.close()
            return

        all_amplitudes = np.array(all_amplitudes, dtype=float)
        std_amplitudes = np.nanstd(all_amplitudes, axis=0)

        # Clear and plot
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.plot(time_points, mean_amplitudes, label='Mean Amplitude', color='purple')
        self.axes.fill_between(time_points, mean_amplitudes - std_amplitudes, mean_amplitudes + std_amplitudes,
                            color='purple', alpha=0.2, label='SD')
        # Baseline files carry negative times, so the treatment starts at zero.
        if files_before_treatment > 0:
            self.axes.axvline(x=0, color='red', linestyle='--', label='Treatment Start')
        self.axes.set_xlabel('Time (minutes)')
        self.axes.set_ylabel('Amplitude')
        self.axes.set_title('Mean Amplitude Over Time (All Experiments)')
        self.axes.legend()
        self.axes.grid(False)
        self.fig.tight_layout()
        self.draw()
        progress.close()

    def show_decay_exponential_fitting(self, group_analysis, replicate_time_point=0):
        """
        Display exponential decay fitting across replicates for a selected time point.

        Args:
            group_analysis (GroupAnalysis): Object managing replicate data and fitting results.
            replicate_time_point (int): Index of the replicate file/time point to fit.

        Returns:
            None
        """
        from PyQt5.QtWidgets import QMessageBox
        # Show loading dialog
        progress = QProgressDialog("Processing data, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()  # Ensure dialog appears

        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)

        try:
            result = group_analysis.exponential_fitting_joint(replicate_time_point)
        except Exception as e:
            # ValueError = mismatched lengths / too few post-peak samples;
            # RuntimeError = curve_fit failed to converge.
            QMessageBox.warning(
                self,
                "Exponential Fit Failed",
                f"Error: {str(e)}\n\nPlease ensure all replicates have the same number of time points."
            )
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)
            self.axes.set_title("Exponential fit failed")
            self.draw()
            progress.close()
            return
        if result is None:
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)
            self.axes.set_title("No data to fit")
            self.draw()
            progress.close()
            return

        from core.group_analysis import draw_exponential_fit

        # Plot on the embedded axes using the SAME drawing routine as the export,
        # so "Save All Plots" produces exactly what is shown here.
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        draw_exponential_fit(self.axes, result, freq)
        self.fig.tight_layout()

        self.draw()
        progress.close()

    def show_tau_param_over_time(self, group_analysis):
        """
        Plot the exponential decay constant (tau) across replicate time points.

        Args:
            group_analysis (GroupAnalysis): Group container with all experiment replicates.

        Returns:
            None
        """
        import numpy as np

        # Show loading dialog
        progress = QProgressDialog("Processing data, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()  # Ensure dialog appears

        tau_list, tau_err_list = group_analysis.get_tau_over_time()
        time_points = group_analysis.get_timepoints_minutes()
        files_before_treatment = group_analysis.get_experiments()[0].get_number_of_files_before_treatment() or 0

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.errorbar(time_points, tau_list, yerr=tau_err_list, fmt='o-', capsize=4, color='C1', label='Tau (decay constant)')
        # Baseline files carry negative times, so the treatment starts at zero.
        if files_before_treatment > 0:
            self.axes.axvline(x=0, color='red', linestyle='--', label='Treatment Start')
        self.axes.set_xlabel("Time (minutes)")
        self.axes.set_ylabel("Tau (decay constant)")
        self.axes.set_title("Exponential Decay Tau Over Time Points")
        self.axes.grid(False)
        self.axes.legend()
        self.fig.tight_layout()
        self.draw()
        progress.close()

    def show_amplitudes_over_time(self, group_analysis):
        """
        Plot raw amplitude traces for each experiment across all time points.

        Args:
            group_analysis (GroupAnalysis): Object holding multiple replicate experiments.

        Returns:
            None
        """
        import numpy as np

        # Show loading dialog
        progress = QProgressDialog("Processing data, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()  # Ensure dialog appears

        time_points, mean_amplitudes, all_amplitudes, files_before_treatment = group_analysis.amplitudes_over_time_all_experiments()
        if time_points is None:
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)
            self.axes.set_title("No data to plot")
            self.draw()
            progress.close()
            return

        all_amplitudes = np.array(all_amplitudes, dtype=float)

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        for i, amplitudes in enumerate(all_amplitudes):
            self.axes.plot(time_points, amplitudes, label=f'Experiment {i+1}', alpha=0.7)
        # Baseline files carry negative times, so the treatment starts at zero.
        if files_before_treatment > 0:
            self.axes.axvline(x=0, color='red', linestyle='--', label='Treatment Start')
        self.axes.set_xlabel('Time (minutes)')
        self.axes.set_ylabel('Amplitude')
        self.axes.set_title('Amplitudes Over Time (All Experiments)')
        self.axes.legend()
        # Ticks must span the negative baseline times too, so anchor them on zero
        # instead of starting at zero.
        from core.group_analysis import GroupAnalysis
        self.axes.set_xticks(GroupAnalysis._time_ticks(time_points))
        self.fig.tight_layout()
        self.draw()
        progress.close()

    def show_spontaneous_peak_frequency(self, group_analysis):
        """
        Plot spontaneous peak frequency over time for all experiments.
        """
        import numpy as np
        from PyQt5.QtCore import QSettings
        from PyQt5.QtWidgets import QProgressDialog, QApplication

        # Show loading dialog
        progress = QProgressDialog("Processing spontaneous peak frequency data...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        try:
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            file_length_sec = settings.value("file_length", 100, type=int)
            time_between_files = settings.value("time_between_files", 10, type=float)

            experiments = group_analysis.get_experiments()
            if not experiments:
                self.fig.clear()
                self.axes = self.fig.add_subplot(111)
                self.axes.set_title("No experiments to analyze")
                self.draw()
                return

            # Collect frequency data from all experiments
            all_frequencies = []
            time_points = []

            n_files = experiments[0].get_file_count()
            files_before_treatment = experiments[0].get_number_of_files_before_treatment()

            # Calculate time points
            if files_before_treatment > 0:
                time_points = [time_between_files * (i - files_before_treatment) for i in range(n_files)]
            else:
                time_points = [i * time_between_files for i in range(n_files)]

            # Collect frequency data for each experiment
            for exp in experiments:
                exp_frequencies = []
                for j, spheroid_file in enumerate(exp.files):
                    meta = spheroid_file.get_metadata()
                    peak_positions = meta.get('peak_amplitude_positions', [])

                    # Handle both single value and list/array cases
                    if not isinstance(peak_positions, (list, np.ndarray)):
                        peak_positions = [peak_positions] if peak_positions else []

                    num_peaks = len(peak_positions)
                    frequency = num_peaks / (file_length_sec / 60)  # peaks per minute
                    exp_frequencies.append(frequency)

                all_frequencies.append(exp_frequencies)

            # Convert to numpy array for easier manipulation
            all_frequencies = np.array(all_frequencies)
            mean_frequencies = np.nanmean(all_frequencies, axis=0)
            std_frequencies = np.nanstd(all_frequencies, axis=0)

            # Clear and plot
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)

            # Plot mean frequency with standard deviation
            self.axes.plot(time_points, mean_frequencies, 'o-', color='#2E8B57',
                          linewidth=2, markersize=6, label='Mean Frequency')
            self.axes.fill_between(time_points,
                                  mean_frequencies - std_frequencies,
                                  mean_frequencies + std_frequencies,
                                  color='#2E8B57', alpha=0.2, label='SD')

            # Plot individual experiments in lighter colors
            for i, frequencies in enumerate(all_frequencies):
                self.axes.plot(time_points, frequencies, '--', alpha=0.5,
                              color='gray', linewidth=1, label=f'Exp {i+1}' if i < 3 else '_nolegend_')

            # Add treatment line if applicable
            if files_before_treatment > 0:
                treatment_time = 0  # Treatment starts at time 0 in this coordinate system
                self.axes.axvline(x=treatment_time, color='red', linestyle='--',
                                 linewidth=2, label='Treatment Start')

            self.axes.set_xlabel('Time (minutes)', fontsize=12)
            self.axes.set_ylabel('Peak Frequency (peaks/min)', fontsize=12)
            self.axes.set_title('Spontaneous Peak Frequency Over Time', fontweight='bold')
            self.axes.legend()
            self.axes.grid(True, alpha=0.3)
            self.fig.tight_layout()
            self.draw()

        except Exception as e:
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)
            self.axes.text(0.5, 0.5, f"Error plotting frequency data: {str(e)}",
                          ha='center', va='center', transform=self.axes.transAxes)
            self.draw()
        finally:
            progress.close()

    def show_spontaneous_peak_amplitudes(self, group_analysis):
        """
        Plot spontaneous peak amplitudes over time for all experiments.
        """
        import numpy as np
        from PyQt5.QtCore import QSettings
        from PyQt5.QtWidgets import QProgressDialog, QApplication

        # Show loading dialog
        progress = QProgressDialog("Processing spontaneous peak amplitude data...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        try:
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            time_between_files = settings.value("time_between_files", 0, type=float)

            experiments = group_analysis.get_experiments()
            if not experiments:
                self.fig.clear()
                self.axes = self.fig.add_subplot(111)
                self.axes.set_title("No experiments to analyze")
                self.draw()
                return

            # Collect amplitude data from all experiments
            all_mean_amplitudes = []
            time_points = []

            n_files = experiments[0].get_file_count()
            files_before_treatment = experiments[0].get_number_of_files_before_treatment()

            # Calculate time points
            if files_before_treatment > 0:
                time_points = [time_between_files * (i - files_before_treatment) for i in range(n_files)]
            else:
                time_points = [i * time_between_files for i in range(n_files)]

            # Collect amplitude data for each experiment
            for exp in experiments:
                exp_mean_amplitudes = []
                for j, spheroid_file in enumerate(exp.files):
                    meta = spheroid_file.get_metadata()
                    peak_values = meta.get('peak_amplitude_values', [])

                    # Handle both single value and list/array cases
                    if not isinstance(peak_values, (list, np.ndarray)):
                        peak_values = [peak_values] if peak_values else []

                    # Calculate mean amplitude for this file
                    mean_amplitude = np.mean(peak_values) if peak_values else 0
                    exp_mean_amplitudes.append(mean_amplitude)

                all_mean_amplitudes.append(exp_mean_amplitudes)

            # Convert to numpy array for easier manipulation
            all_mean_amplitudes = np.array(all_mean_amplitudes)
            mean_amplitudes = np.nanmean(all_mean_amplitudes, axis=0)
            std_amplitudes = np.nanstd(all_mean_amplitudes, axis=0)

            # Clear and plot
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)

            # Plot mean amplitude with standard deviation
            self.axes.plot(time_points, mean_amplitudes, 'o-', color='#8B008B',
                          linewidth=2, markersize=6, label='Mean Amplitude')
            self.axes.fill_between(time_points,
                                  mean_amplitudes - std_amplitudes,
                                  mean_amplitudes + std_amplitudes,
                                  color='#8B008B', alpha=0.2, label='SD')

            # Plot individual experiments in lighter colors
            for i, amplitudes in enumerate(all_mean_amplitudes):
                self.axes.plot(time_points, amplitudes, '--', alpha=0.5,
                              color='gray', linewidth=1, label=f'Exp {i+1}' if i < 3 else '_nolegend_')

            # Add treatment line if applicable
            if files_before_treatment > 0:
                treatment_time = 0  # Treatment starts at time 0 in this coordinate system
                self.axes.axvline(x=treatment_time, color='red', linestyle='--',
                                 linewidth=2, label='Treatment Start')

            self.axes.set_xlabel('Time (minutes)', fontsize=12)
            self.axes.set_ylabel('Mean Peak Amplitude (nA)', fontsize=12)
            self.axes.set_title('Spontaneous Peak Amplitudes Over Time', fontweight='bold')
            self.axes.legend()
            self.axes.grid(True, alpha=0.3)
            self.fig.tight_layout()
            self.draw()

        except Exception as e:
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)
            self.axes.text(0.5, 0.5, f"Error plotting amplitude data: {str(e)}",
                          ha='center', va='center', transform=self.axes.transAxes)
            self.draw()
        finally:
            progress.close()

    @staticmethod
    def collect_peak_times(group_analysis):
        """
        Gather every detected peak on a single experiment-wide time axis.

        Each peak's absolute time combines where its file sits in the experiment
        (``time_between_files`` apart, negative before treatment) with where the
        peak sits inside that file (sample index / acquisition frequency).

        Args:
            group_analysis: GroupAnalysis instance with processed experiments.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: one (times_sec, amplitudes) pair
            per experiment. Experiments with no peaks yield empty arrays.
        """
        import numpy as np

        per_experiment = []
        for exp in group_analysis.get_experiments():
            freq = exp.get_acquisition_frequency() or 1
            interval_sec = exp.get_time_between_files() * 60.0
            files_before = exp.get_number_of_files_before_treatment()

            times, amps = [], []
            for file_index, spheroid_file in enumerate(exp.files):
                meta = spheroid_file.get_metadata()
                positions = meta.get('peak_amplitude_positions', [])
                values = meta.get('peak_amplitude_values', [])

                # Single-peak files store scalars rather than lists.
                if not isinstance(positions, (list, tuple, np.ndarray)):
                    positions = [positions] if positions is not None else []
                if not isinstance(values, (list, tuple, np.ndarray)):
                    values = [values] if values is not None else []

                file_start_sec = (file_index - files_before) * interval_sec
                for pos, val in zip(positions, values):
                    if pos is None or val is None:
                        continue
                    times.append(file_start_sec + float(pos) / freq)
                    amps.append(float(val))

            per_experiment.append((np.asarray(times, dtype=float),
                                   np.asarray(amps, dtype=float)))
        return per_experiment

    def show_windowed_peak_amplitudes(self, group_analysis, window_sec=60.0):
        """
        Plot mean peak amplitude per time window across the whole experiment.

        Peaks are placed on an absolute time axis (file position + position
        within the file), then bucketed into consecutive windows of
        ``window_sec``. Every peak falling in a window contributes to that
        window's mean amplitude, so a 60 s window containing peaks at 10 s and
        30 s reports the mean of those two.

        Args:
            group_analysis: GroupAnalysis instance with processed experiments.
            window_sec (float): Width of the averaging window in seconds.
        """
        import numpy as np
        from PyQt5.QtWidgets import QProgressDialog, QApplication

        progress = QProgressDialog("Binning peak amplitudes...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        try:
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)

            experiments = group_analysis.get_experiments()
            if not experiments:
                self.axes.set_title("No experiments to analyze")
                self.draw()
                return

            window_sec = float(window_sec) if window_sec and window_sec > 0 else 60.0
            per_experiment = self.collect_peak_times(group_analysis)

            all_times = np.concatenate([t for t, _a in per_experiment if t.size]) \
                if any(t.size for t, _a in per_experiment) else np.array([])
            if all_times.size == 0:
                self.axes.text(0.5, 0.5, "No peaks were detected in these files.",
                               ha='center', va='center', transform=self.axes.transAxes)
                self.draw()
                return

            # Window edges anchored on 0 (treatment start) so baseline windows
            # fall on negative multiples of the window width.
            first_edge = np.floor(all_times.min() / window_sec) * window_sec
            last_edge = (np.floor(all_times.max() / window_sec) + 1) * window_sec
            edges = np.arange(first_edge, last_edge + window_sec / 2, window_sec)
            centers_min = (edges[:-1] + window_sec / 2) / 60.0

            # Pooled statistics across all replicates, per window.
            pooled_times = all_times
            pooled_amps = np.concatenate([a for _t, a in per_experiment if a.size])
            bin_idx = np.clip(np.digitize(pooled_times, edges) - 1, 0, len(edges) - 2)

            means, sds, counts = [], [], []
            for b in range(len(edges) - 1):
                vals = pooled_amps[bin_idx == b]
                counts.append(vals.size)
                means.append(np.mean(vals) if vals.size else np.nan)
                sds.append(np.std(vals) if vals.size else np.nan)
            means = np.asarray(means)
            sds = np.asarray(sds)
            counts = np.asarray(counts)

            filled = counts > 0

            # Individual replicates in the background
            for i, (times, amps) in enumerate(per_experiment):
                if times.size == 0:
                    continue
                idx = np.clip(np.digitize(times, edges) - 1, 0, len(edges) - 2)
                exp_bins = np.array([b for b in range(len(edges) - 1) if np.any(idx == b)])
                if exp_bins.size == 0:
                    continue
                exp_means = np.array([np.mean(amps[idx == b]) for b in exp_bins])
                # Windows with no peaks are skipped rather than drawn as gaps, so
                # the replicate traces stay comparable with the pooled mean.
                self.axes.plot(centers_min[exp_bins], exp_means, 'o--', color='gray',
                               alpha=0.45, markersize=3, linewidth=1,
                               label=f'Exp {i + 1}' if i < 3 else '_nolegend_')

            self.axes.errorbar(centers_min[filled], means[filled], yerr=sds[filled],
                               fmt='o-', color='#8B008B', ecolor='#8B008B',
                               elinewidth=1, capsize=3, linewidth=2, markersize=6,
                               label=f'Mean Amplitude ({window_sec:g} s windows)')

            # Peak counts per window, only while they stay readable.
            if filled.sum() <= 25:
                for x, y, n in zip(centers_min[filled], means[filled], counts[filled]):
                    self.axes.annotate(f'n={n}', (x, y), textcoords='offset points',
                                       xytext=(0, 8), ha='center', fontsize=8, color='dimgray')

            if experiments[0].get_number_of_files_before_treatment() > 0:
                self.axes.axvline(x=0, color='red', linestyle='--', linewidth=2,
                                  label='Treatment Start')

            # One window per point crowds the axis, so label only every few
            # minutes instead of letting matplotlib tick each window.
            from core.group_analysis import GroupAnalysis
            self.axes.set_xticks(GroupAnalysis._time_ticks(centers_min[filled]))

            self.axes.set_xlabel('Time (minutes)', fontsize=12)
            self.axes.set_ylabel('Mean Peak Amplitude (nA)', fontsize=12)
            self.axes.set_title(
                f'Peak Amplitudes in {window_sec:g} s Windows '
                f'({int(counts.sum())} peaks, {int(filled.sum())} windows)',
                fontweight='bold')
            self.axes.legend()
            self.axes.grid(True, alpha=0.3)
            self.fig.tight_layout()
            self.draw()

        except Exception as e:
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)
            self.axes.text(0.5, 0.5, f"Error plotting windowed amplitudes: {str(e)}",
                           ha='center', va='center', transform=self.axes.transAxes)
            self.draw()
        finally:
            progress.close()

    def plot_cv(self, processed_data, time_point=0, metadata=None, title_suffix=None, ref_voltage=None):
        """
        Enhanced CV plot method that can handle both single time point and peak-correlated CVs.
        """
        # If metadata contains peak information, use the peak-correlated version
        if metadata and 'peak_amplitude_positions' in metadata:
            self.plot_cv_multiple_peaks(processed_data, metadata, title_suffix, ref_voltage=ref_voltage)
        else:
            # Fallback to original single CV plot
            self.plot_cv_single(processed_data, time_point, title_suffix, ref_voltage=ref_voltage)


    def plot_cv_multiple_peaks(self, processed_data, metadata=None, title_suffix=None, ref_voltage=None):
        """
        Plot cyclic voltammograms (CVs) at time points corresponding to detected peaks in I-T curve.
        Shows multiple CVs overlayed if multiple peaks are detected.

        Parameters:
        - processed_data: 2D numpy array
        - metadata: dict containing peak information from I-T analysis
        - title_suffix: optional suffix for the title
        """
        from core.spheroid_file import voltage_axis_for_waveform

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

        # Generate voltage waveform based on the configured waveform type.
        # None means the waveform has no defined sweep ("Non-specific"), in which
        # case the scan index is used as the x-axis.
        waveform_type = QSettings("HashemiLab", "NeuroStemVolt").value("waveform", "5HT", type=str)
        voltage = voltage_axis_for_waveform(waveform_type, processed_data.shape[1])
        if voltage is None:
            voltage = np.arange(processed_data.shape[1])

        # Get peak time points from metadata
        peak_time_points = []
        if metadata and 'peak_amplitude_positions' in metadata:
            peak_indices = metadata['peak_amplitude_positions']

            # Handle both single peak and multiple peaks
            if isinstance(peak_indices, (list, np.ndarray)):
                peak_time_points = [int(idx) for idx in peak_indices if 0 <= idx < processed_data.shape[0]]
            elif isinstance(peak_indices, (int, np.integer)):
                if 0 <= peak_indices < processed_data.shape[0]:
                    peak_time_points = [int(peak_indices)]

        # If no peaks detected, use middle time point as fallback
        if not peak_time_points:
            peak_time_points = [processed_data.shape[0] // 2]

        # Colors for different CVs
        colors = ['#4178F2', '#FF3877', '#32CD32', '#FF8C00', '#9370DB', '#00CED1', '#FFD700', '#DC143C']

        # Plot CV for each peak time point
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)

        max_current = 0
        for i, time_point in enumerate(peak_time_points[:5]):  # Limit to first 5 peaks for clarity
            # Extract current values for this time point
            current = processed_data[time_point, :]

            # Find the peak current and corresponding voltage for this CV
            peak_i = np.argmax(current)
            peak_V = voltage[peak_i]
            peak_I = current[peak_i]

            # Track max current for scaling
            max_current = max(max_current, np.max(current))

            # Plot the CV
            color = colors[i % len(colors)]
            time_sec = time_point / freq

            if len(peak_time_points) == 1:
                label = f"CV at {time_sec:.1f}s"
            else:
                label = f"CV {i + 1} at {time_sec:.1f}s"

            self.axes.plot(voltage, current, color=color, linewidth=1.5, label=label, alpha=0.8)

            # Mark the peak on each CV
            self.axes.scatter([peak_V], [peak_I], color=color, s=80, zorder=5, alpha=0.9)

            # Annotate only the first CV or if there's only one CV to avoid clutter
            if i == 0 or len(peak_time_points) == 1:
                self.axes.annotate(f"Peak: {peak_V:.2f} V\n{peak_I:.2f} nA",
                                   xy=(peak_V, peak_I),
                                   xytext=(0, -42), textcoords='offset points',
                                   ha='center',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                             edgecolor='#888888', alpha=0.9),
                                   arrowprops=dict(arrowstyle='->', color=color),
                                   fontsize=8)

        # Axis labels and formatting
        self.axes.set_xlabel("Voltage (V)")
        self.axes.set_ylabel("Current (nA)")

        # Title based on number of peaks
        if len(peak_time_points) == 1:
            title = f"Cyclic Voltammogram at Peak Time"
        else:
            title = f"Cyclic Voltammograms at {len(peak_time_points)} Peak Times"

        if title_suffix:
            title += f" - {title_suffix}"

        self.axes.set_title(title, fontweight="bold")
        self.axes.grid(True, alpha=0.3)

        # Only show legend if there are multiple CVs
        if len(peak_time_points) > 1:
            self.axes.legend(fontsize=8, loc='upper right')

        # Add info box showing peak correlation
        info_text = f"Peaks detected: {len(peak_time_points)}"
        if metadata and 'num_peaks_detected' in metadata:
            info_text = f"I-T Peaks: {metadata['num_peaks_detected']}"

        self.axes.text(0.02, 0.98, info_text, transform=self.axes.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Optional reference voltage line
        if ref_voltage is not None:
            self.axes.axvline(x=ref_voltage, color='#e74c3c', linestyle='--',
                              linewidth=1.5, alpha=0.85, label=f'Ref: {ref_voltage:.2f} V')
            self.axes.legend(fontsize=8, loc='upper right')

        self.fig.tight_layout()
        self.draw()

    def plot_cv_single(self, processed_data, time_point=0, title_suffix=None, ref_voltage=None):
        """
        Original single CV plot method (renamed for clarity).
        """
        from core.spheroid_file import voltage_axis_for_waveform

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

        # Generate voltage waveform based on the configured waveform type.
        # None means the waveform has no defined sweep ("Non-specific"), in which
        # case the scan index is used as the x-axis.
        waveform_type = QSettings("HashemiLab", "NeuroStemVolt").value("waveform", "5HT", type=str)
        voltage = voltage_axis_for_waveform(waveform_type, processed_data.shape[1])
        if voltage is None:
            voltage = np.arange(processed_data.shape[1])

        # Extract current values for the requested CV (time point)
        if time_point >= processed_data.shape[0]:
            time_point = processed_data.shape[0] - 1

        current = processed_data[time_point, :]

        # Find the peak current and corresponding voltage
        peak_i = np.argmax(current)
        peak_V = voltage[peak_i]
        peak_I = current[peak_i]

        # Plot the CV
        self.axes.plot(voltage, current, label="CV", color="#4178F2", linewidth=1.5)
        self.axes.scatter([peak_V], [peak_I], color="#FF3877", s=100, zorder=5, label='Peak')
        self.axes.annotate(f"Peak: {peak_V:.2f} V\n{peak_I:.2f} nA",
                           xy=(peak_V, peak_I),
                           xytext=(0, -42), textcoords='offset points',
                           ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                     edgecolor='#888888', alpha=0.9),
                           arrowprops=dict(arrowstyle='->', color='red'))

        x_label = "Scan Index" if waveform_type == "Non-specific" else "Voltage (V)"
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel("Current (nA)")

        # Convert time_point to seconds for display
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)
        time_sec = time_point / freq

        title = f"Cyclic Voltammogram at {time_sec:.1f}s"
        if title_suffix:
            title += f" - {title_suffix}"

        self.axes.set_title(title, fontweight="bold")
        self.axes.grid(True, alpha=0.3)

        # Optional reference voltage line
        if ref_voltage is not None:
            self.axes.axvline(x=ref_voltage, color='#e74c3c', linestyle='--',
                              linewidth=1.5, alpha=0.85, label=f'Ref: {ref_voltage:.2f} V')

        self.axes.legend()
        self.fig.tight_layout()
        self.draw()

