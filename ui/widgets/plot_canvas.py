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

    def plot_color(self, processed_data, peak_pos = None, title_suffix=None):
        """
        Display a 2D color plot of the FSCV data matrix.

        Args:
            processed_data (np.ndarray): 2D data array (voltage x time).
            peak_pos (int, optional): Index of the peak position for annotation.
            title_suffix (str, optional): Text to append to the plot title.

        Returns:
            None
        """
        from core.spheroid_file import PLOT_SETTINGS  # Ensure it's correctly imported

        plot_settings = PLOT_SETTINGS()
        custom_cmap = plot_settings.custom

        vmin = np.percentile(processed_data, 1)
        vmax = np.percentile(processed_data, 99)

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

        im = self.axes.imshow(
            processed_data.T,       # Transpose to align voltage steps as rows
            aspect='auto',
            cmap=custom_cmap,
            origin='lower',
            extent=[0, processed_data.shape[0], 0, processed_data.shape[1]],
            vmin=vmin,
            vmax=vmax
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

        self.axes.set_xlabel("Time Points")
        self.axes.set_ylabel("Voltage Steps")
        title = f"Color Plot{': ' + title_suffix if title_suffix else ''}"
        self.axes.set_title(title, fontweight='bold')

        self.fig.tight_layout()
        self.draw()

    def plot_IT(self, processed_data, metadata=None, peak_position=None, temp_peak_detection=None):
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
        self.axes.clear()

        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)

        self.axes = self.fig.add_subplot(111)
        profile = processed_data[:, peak_position]
        n = profile.shape[0]
        t = np.arange(n) / freq  # in seconds

        # Plot the main profile
        self.axes.plot(t, profile, label="I-T Profile", color='#4178F2', linewidth=1.5)

        # Plot peak markers if metadata is provided
        if metadata and 'peak_amplitude_positions' in metadata:
            print("IN PLOT CANVAS")
            print(metadata)
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
            self.axes.clear()
            self.axes.set_title("No data to plot")
            self.draw()
            progress.close()
            return

        all_amplitudes = np.array(all_amplitudes, dtype=float)
        std_amplitudes = np.nanstd(all_amplitudes, axis=0)

        # Clear and plot
        self.axes.clear()
        self.axes.plot(time_points, mean_amplitudes, label='Mean Amplitude', color='purple')
        self.axes.fill_between(time_points, mean_amplitudes - std_amplitudes, mean_amplitudes + std_amplitudes,
                            color='purple', alpha=0.2, label='SD')
        if files_before_treatment > 0:
            treatment_time = files_before_treatment * (time_points[1] - time_points[0])
            self.axes.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')
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
            result = group_analysis.exponential_fitting_replicated(replicate_time_point)
        except ValueError as e:
            QMessageBox.warning(
                self,
                "Dimension Mismatch",
                f"Error: {str(e)}\n\nPlease ensure all replicates have the same number of time points."
            )
            self.axes.clear()
            self.axes.set_title("Dimension mismatch error")
            self.draw()
            progress.close()
            return
        if result is None:
            self.axes.clear()
            self.axes.set_title("No data to fit")
            self.draw()
            progress.close()
            return

        from scipy.stats import t
        import numpy as np

        # Get fit and aligned ITs from group_analysis
        # result = group_analysis.exponential_fitting_replicated(replicate_time_point)
        time_all, cropped_ITs, _, t_half, fit_vals, fit_errs, min_peak = result

        A_fit, k_fit, C_fit = fit_vals
        A_err, k_err, C_err = fit_errs
        tau_fit = 1 / k_fit if k_fit != 0 else np.nan

        n_exps, n_post = cropped_ITs.shape
        t_rel = np.arange(n_post) / freq        # seconds

        mean_IT = np.nanmean(cropped_ITs, axis=0)
        std_IT  = np.nanstd (cropped_ITs, axis=0)
        # do the fit in sample‐point space, then map to seconds:
        t_fit_pts = np.linspace(0, n_post-1, 500)          # in points
        y_fit     = (A_fit - C_fit) * np.exp(-k_fit * t_fit_pts) + C_fit
        t_fit_rel = t_fit_pts / freq                       # now in seconds

        # 95% CI of the fit via Jacobian
        dof  = max(0, len(time_all) - 3)
        tval = t.ppf(0.975, dof)
        J        = np.empty((len(t_fit_rel), 3))
        J[:, 0] = np.exp(-t_fit_rel * k_fit)
        J[:, 1] = -(A_fit - C_fit) * t_fit_rel * np.exp(-t_fit_rel * k_fit)
        J[:, 2] = 1 - np.exp(-t_fit_rel * k_fit)
        pcov = np.diag([A_err**2, k_err**2, C_err**2])
        ci = np.sqrt(np.sum((J @ pcov) * J, axis=1)) * tval
        lower_ci = y_fit - ci
        upper_ci = y_fit + ci

        # Plot on the embedded axes
        self.axes.clear()

        # a) each replicate in light gray
        for row in cropped_ITs:
            self.axes.plot(t_rel, row, color='gray', alpha=0.3, lw=1, label='_nolegend_')

        # b) individual data points used for fitting
        ITs_flattened = cropped_ITs.flatten()
        t_data = (np.array(time_all)) / freq
        self.axes.scatter(t_data, ITs_flattened, color='black', s=16, alpha=0.7, label='Data points')

        # d) fitted exponential curve
        self.axes.plot(t_fit_rel, y_fit, color='C1', lw=2, label='Exp fit')

        # e) 95% CI around the fit
        self.axes.fill_between(t_fit_rel, lower_ci, upper_ci, color='C1', alpha=0.3, label='95% CI')

        # f) half-life marker
        t_half_s = t_half / freq
        self.axes.axvline(t_half_s, color='magenta', ls='--',label=f't½ ≈ {t_half_s:.2f} s')

        # 7) labels & styling
        self.axes.set_xlabel('Time (seconds)', fontsize=12)
        self.axes.set_ylabel('Current (nA)', fontsize=12)
        self.axes.set_title('Post-peak IT decays & exponential fit', fontsize=14)
        self.axes.legend(frameon=False)
        self.axes.grid(False)
        self.fig.tight_layout()

        max_t = t_rel[-1]  # since t_rel is in seconds

        tick_interval = 5  # seconds
        ticks = np.arange(0, max_t + tick_interval, tick_interval)
        self.axes.set_xticks(ticks)

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
        n_files = group_analysis.get_experiments()[0].get_file_count()
        time_points = np.linspace(
            0,
            group_analysis.get_experiments()[0].get_time_between_files() * (n_files - 1),
            n_files
        )

        self.axes.clear()
        self.axes.errorbar(time_points, tau_list, yerr=tau_err_list, fmt='o-', capsize=4, color='C1', label='Tau (decay constant)')
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
            self.axes.clear()
            self.axes.set_title("No data to plot")
            self.draw()
            progress.close()
            return

        all_amplitudes = np.array(all_amplitudes, dtype=float)
        treatment_time = files_before_treatment * (time_points[1] - time_points[0])

        self.axes.clear()
        for i, amplitudes in enumerate(all_amplitudes):
            self.axes.plot(time_points, amplitudes, label=f'Experiment {i+1}', alpha=0.7)
        if files_before_treatment > 0:
            self.axes.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')
        self.axes.set_xlabel('Time (min)')
        self.axes.set_ylabel('Amplitude')
        self.axes.set_title('Amplitudes Over Time (All Experiments)')
        self.axes.legend()
        max_t = time_points[-1]
        tick_interval = 5
        self.axes.set_xticks(np.arange(0, max_t + tick_interval, tick_interval))
        self.fig.tight_layout()
        self.draw()
        progress.close()

    def show_frequency_over_time(self, group_analysis):
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

        time_points, mean_frequency, all_frequencies, files_before_treatment = group_analysis.frequency_over_time_all_experiments()

        if time_points is None:
            self.axes.clear()
            self.axes.set_title("No data to plot")
            self.draw()
            progress.close()
            return

        all_frequencies = np.array(all_frequencies, dtype=float)
        treatment_time = files_before_treatment * (time_points[1] - time_points[0])

        self.axes.clear()
        for i, frequencies in enumerate(all_frequencies):
            self.axes.plot(time_points, frequencies, label=f'Experiment {i+1}', alpha=0.7)
        if files_before_treatment > 0:
            self.axes.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')
        self.axes.set_xlabel('Time (min)')
        self.axes.set_ylabel('Frequency (Hz)')
        self.axes.set_title('Frequency Over Time (All Experiments)')
        self.axes.legend()
        max_t = time_points[-1]
        tick_interval = 5
        self.axes.set_xticks(np.arange(0, max_t + tick_interval, tick_interval))
        self.fig.tight_layout()
        self.draw()
        progress.close()

    def show_amplitude_for_single_file(self, group_analysis, replicate_time_point=0):
        """
        Plot amplitude time series (per second) within a single selected file (time point).
        Caps the x-axis at 60 s or the file duration, whichever is shorter.

        Expects backend to provide per-second amplitudes via
        `group_analysis.amplitude_series_for_file_all_experiments(idx, bin_seconds=1)`.
        Falls back with a warning if unavailable.
        """
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox

        # Loading dialog
        progress = QProgressDialog("Processing data, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        # Try to get per-second amplitude series from backend
        try:
            fetcher = getattr(group_analysis, 'amplitude_series_for_file_all_experiments')
        except AttributeError:
            progress.close()
            QMessageBox.warning(self, "Not Implemented",
                                "Backend is missing 'amplitude_series_for_file_all_experiments'.\n"
                                "Please implement it to return (t_seconds, amplitudes_2d).")
            return

        try:
            t_seconds, amp_2d = fetcher(int(replicate_time_point), bin_seconds=1)
        except Exception as e:
            progress.close()
            QMessageBox.warning(self, "Error", f"Could not compute per-second amplitudes for this file.\n{e}")
            return

        if t_seconds is None or amp_2d is None:
            self.axes.clear()
            self.axes.set_title("No data to plot")
            self.draw(); progress.close()
            return

        t_seconds = np.asarray(t_seconds, dtype=float)
        amp_2d = np.asarray(amp_2d, dtype=float)
        if amp_2d.ndim == 1:
            amp_2d = amp_2d[None, :]
        n_exps, n_t = amp_2d.shape
        if n_t == 0:
            self.axes.clear(); self.axes.set_title("No time samples available"); self.draw(); progress.close(); return

        # Cap duration at 60 seconds or the available duration
        cap_T = 60.0 if t_seconds[-1] >= 60.0 else t_seconds[-1]
        mask = t_seconds <= cap_T
        t_plot = t_seconds[mask]
        amp_plot = amp_2d[:, mask]

        # Plot lines (per experiment)
        self.axes.clear()
        for i in range(n_exps):
            self.axes.plot(t_plot, amp_plot[i], alpha=0.9, linewidth=1.5, label=f"Experiment {i+1}")

        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Amplitude')
        self.axes.set_title('Amplitude Within File (per second)')
        self.axes.grid(False)
        if n_exps > 1:
            self.axes.legend()

        # Ticks every 5 seconds for readability
        max_t = t_plot[-1]
        tick_interval = 5
        self.axes.set_xticks(np.arange(0, max_t + tick_interval, tick_interval))

        self.fig.tight_layout()
        self.draw()
        progress.close()

    def show_amplitudes_for_timepoint(self, group_analysis, replicate_time_point=0):
        """
        Compatibility alias: same as show_amplitude_for_single_file.
        """
        return self.show_amplitude_for_single_file(group_analysis, replicate_time_point)

    def plot_cv(self, processed_data, time_point=0, metadata=None, title_suffix=None):
        """
        Enhanced CV plot method that can handle both single time point and peak-correlated CVs.
        """
        # If metadata contains peak information, use the peak-correlated version
        if metadata and 'peak_amplitude_positions' in metadata:
            self.plot_cv_multiple_peaks(processed_data, metadata, title_suffix)
        else:
            # Fallback to original single CV plot
            self.plot_cv_single(processed_data, time_point, title_suffix)


    def plot_cv_multiple_peaks(self, processed_data, metadata=None, title_suffix=None):
        """
        Plot cyclic voltammograms (CVs) at time points corresponding to detected peaks in I-T curve.
        Shows multiple CVs overlayed if multiple peaks are detected.

        Parameters:
        - processed_data: 2D numpy array
        - metadata: dict containing peak information from I-T analysis
        - title_suffix: optional suffix for the title
        """
        from core.spheroid_file import Waveforms

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

        # Generate voltage waveform (assuming 5HT waveform parameters)
        waveform_type = QSettings("HashemiLab", "NeuroStemVolt").value("waveform", "None", type=str)
        try:
            if waveform_type == "5HT":
                wf = Waveforms(0.2, [1.0, -0.1], 0.2, 1000, processed_data.shape[1])
            else:
                wf = Waveforms(-0.5, [-0.7, 1.1], -0.5, 600, processed_data.shape[1])
            voltage = wf.voltage_waveform()
        except:
            # Fallback: create a simple linear voltage sweep if Waveforms fails
            voltage = np.linspace(-0.1, 1.0, processed_data.shape[1])

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
                self.axes.annotate(f"Peak: {peak_V:.2f}V\n{peak_I:.2f}nA",
                                   xy=(peak_V, peak_I),
                                   xytext=(10, 10), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
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

        self.fig.tight_layout()
        self.draw()

    def plot_cv_single(self, processed_data, time_point=0, title_suffix=None):
        """
        Original single CV plot method (renamed for clarity).
        """
        from core.spheroid_file import Waveforms

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

        waveform_type = QSettings("HashemiLab", "NeuroStemVolt").value("waveform", "None", type=str)

        # Generate voltage waveform (assuming 5HT waveform parameters)
        try:
            if waveform_type == "5HT":
                wf = Waveforms(0.2, [1.0, -0.1], 0.2, 1000, processed_data.shape[1])
            else:
                wf = Waveforms(-0.5, [-0.7, 1.1], -0.5, 600, processed_data.shape[1])
            voltage = wf.voltage_waveform()
        except:
            # Fallback: create a simple linear voltage sweep if Waveforms fails
            voltage = np.linspace(-0.1, 1.0, processed_data.shape[1])

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
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='red'))

        self.axes.set_xlabel("Voltage (V)")
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
        self.axes.legend()

        self.fig.tight_layout()
        self.draw()

