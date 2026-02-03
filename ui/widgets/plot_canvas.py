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

    def _clear_figure(self):
        """
        Clear the figure and remove any existing colorbar.
        This ensures colorbars don't persist between different plot types.
        """
        # Remove colorbar if it exists
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        # Clear the entire figure
        self.fig.clear()
        # Recreate the axes
        self.axes = self.fig.add_subplot(111)

    def plot_color(self, processed_data, peak_pos = None, title_suffix=None):
        """
        Display a 2D color plot of the FSCV data matrix.

        Args:
            processed_data (np.ndarray): 2D data array (voltage x time).
            peak_pos (int, optional): Index of the peak position for annotation.
            metadata (dict, optional): Metadata containing acquisition frequency.
            title_suffix (str, optional): Text to append to the plot title.

        Returns:
            None
        """
        from core.spheroid_file import PLOT_SETTINGS
        plot_settings = PLOT_SETTINGS()
        custom_cmap = plot_settings.custom

        vmin = np.percentile(processed_data, 1)
        vmax = np.percentile(processed_data, 99)

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

        # Get acquisition frequency for time axis conversion
        # Prefer per-file frequency from metadata; fallback to global setting
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        if metadata and 'acquisition_frequency' in metadata:
            try:
                freq = float(metadata['acquisition_frequency'])
            except Exception:
                freq = settings.value("acquisition_frequency", 10, type=int)
        else:
            freq = settings.value("acquisition_frequency", 10, type=int)

        time_extent_seconds = processed_data.shape[0] / freq

        im = self.axes.imshow(
            processed_data.T,       # Transpose to align voltage steps as rows
            aspect='auto',
            cmap=custom_cmap,
            origin='lower',
            extent=[0, time_extent_seconds, 0, processed_data.shape[1]],
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

        self.axes.set_xlabel("Time (seconds)")
        self.axes.set_ylabel("Voltage Steps")
        title = f"Color Plot{': ' + title_suffix if title_suffix else ''}"
        self.axes.set_title(title, fontweight='bold')

        self.fig.tight_layout()
        self.draw()

    def plot_IT(self, processed_data, metadata=None, peak_position=None, temp_peak_detection=None, show_peak_detection=False):
        """
        Plot the I-T (Current vs Time) trace, with optional peak annotations.

        Args:
            processed_data (np.ndarray): 2D data array (voltage x time).
            metadata (dict, optional): Dictionary with peak positions/values.
            peak_position (int, optional): Voltage index used for I-T extraction.
            show_peak_detection (bool, optional): If True, visualize peak detection algorithm with rise/decay windows.

        Returns:
            None
        """
        self.fig.clear()
        self.axes.clear()

        # Prefer per-file frequency from metadata; fallback to global setting
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        if metadata and 'acquisition_frequency' in metadata:
            try:
                freq = float(metadata['acquisition_frequency'])
            except Exception:
                freq = settings.value("acquisition_frequency", 10, type=int)
        else:
            freq = settings.value("acquisition_frequency", 10, type=int)

        
        # Check if calibration is enabled
        calibration_enabled = settings.value("calibration_enabled", False, type=bool)

        self.axes = self.fig.add_subplot(111)
        profile = processed_data[:, peak_position]
        n = profile.shape[0]
        t = np.arange(n) / freq  # in seconds

        # Plot the main profile
        self.axes.plot(t, profile, label="Profile", color='#4178F2', linewidth=1.5)

        # Plot peak markers if metadata is provided
        if metadata and 'peak_amplitude_positions' in metadata:
            print("IN PLOT CANVAS")
            print(f"Metadata: {metadata}")
            print(f"freq used: {freq}")
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
                    # Ensure idx is an integer and within bounds
                    try:
                        idx = int(idx)
                    except (ValueError, TypeError):
                        continue
                    
                    if 0 <= idx < len(profile) and 0 <= idx < len(t):
                        color = colors[i % len(colors)]  # Cycle through colors
                        peak_time = t[idx]  # Convert index to seconds
                        print(f"Peak {i+1}: idx={idx}, peak_time={peak_time:.3f}s, val={val:.3f}")
                        self.axes.scatter(peak_time, val, color=color, s=100, zorder=5,
                                          label=f'Peak {i + 1}' if i < 5 else None)  # Limit legend entries

                        # Annotate with peak number and value
                        self.axes.annotate(f"P{i + 1}: {val:.2f}", (peak_time, val),
                                           textcoords="offset points", xytext=(0, 15),
                                           ha='center', fontsize=8, color=color, fontweight='bold')

                # Add summary text (moved outside loop)
                num_peaks = len(peak_indices)
                self.axes.text(0.02, 0.98, f"Detected {num_peaks} peaks",
                               transform=self.axes.transAxes, fontsize=10,
                               verticalalignment='top', bbox=dict(boxstyle='round',
                                                                  facecolor='wheat', alpha=0.8))

            else:
                # Single peak (backward compatibility)
                try:
                    idx = int(peak_indices) if not isinstance(peak_indices, (list, np.ndarray)) else int(peak_indices[0])
                    val = float(peak_values) if not isinstance(peak_values, (list, np.ndarray)) else float(peak_values[0])
                    if 0 <= idx < len(profile) and 0 <= idx < len(t):
                        peak_time = t[idx]
                        self.axes.scatter(peak_time, val, color='#FF3877', s=100, alpha=0.5, zorder=5, label='Current Peak')
                        self.axes.annotate(f"{val:.2f}", (peak_time, val), textcoords="offset points",
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
        
        # Visualize peak detection algorithm if requested
        if show_peak_detection and metadata and 'peak_amplitude_positions' in metadata and not is_buffer_file:
            peak_indices = metadata['peak_amplitude_positions']
            
            # Handle single or multiple peaks
            if not isinstance(peak_indices, (list, np.ndarray)):
                peak_indices = [peak_indices]
            
            # Get validation parameters if available
            validation_params = metadata.get('decay_validation_params', {})
            
            if validation_params and 'rise_window_samples' in validation_params:
                rise_samples = validation_params.get('rise_window_samples', 20)
                decay_samples = validation_params.get('decay_window_samples', 400)
                
                # Visualize for the first peak (most prominent)
                if len(peak_indices) > 0:
                    peak_idx = int(peak_indices[0])
                    
                    if 0 <= peak_idx < len(profile):
                        # Rise window (before peak)
                        rise_start = max(0, peak_idx - rise_samples)
                        rise_end = peak_idx
                        self.axes.axvspan(t[rise_start], t[rise_end], 
                                        alpha=0.2, color='green', 
                                        label='Rise Window')
                        
                        # Decay window (after peak)
                        decay_start = peak_idx
                        decay_end = min(len(profile) - 1, peak_idx + decay_samples)
                        self.axes.axvspan(t[decay_start], t[decay_end], 
                                        alpha=0.2, color='orange', 
                                        label='Decay Window')
                        
                        # Add annotation showing window durations
                        rise_time = rise_samples / freq
                        decay_time = decay_samples / freq
                        annotation_text = f"Rise: {rise_time:.1f}s | Decay: {decay_time:.1f}s"
                        self.axes.text(0.5, 0.02, annotation_text,
                                     transform=self.axes.transAxes,
                                     fontsize=9, ha='center',
                                     bbox=dict(boxstyle='round', facecolor='white', 
                                             edgecolor='gray', alpha=0.9))

        # Axis labeling and formatting
        self.axes.set_xlabel("Time (seconds)")
        # Set y-axis label based on calibration
        y_label = "Concentration (nM)" if calibration_enabled else "Current (nA)"
        self.axes.set_ylabel(y_label)
        
        # Update title to reflect what's being shown
        trace_type = "C-T" if calibration_enabled else "I-T"
        title = f"{trace_type} Profile"
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
        calibration_enabled = settings.value("calibration_enabled", False, type=bool)
        #self.axes.set_ylabel('Current (nA)', fontsize=12)
        self.axes.set_ylabel("Peak Concentration (nM)" if calibration_enabled else "Peak Amplitude (nA)")
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
        
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        calibration_enabled = settings.value("calibration_enabled", False, type=bool)

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
        y_label = "Peak Concentration (nM)" if calibration_enabled else "Peak Amplitude (nA)"
        self.axes.set_ylabel(y_label)
        
        title_prefix = "Concentrations" if calibration_enabled else "Amplitudes"
        self.axes.set_title(f'{title_prefix} Over Time (All Experiments)')
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

        self._clear_figure()

        # Generate voltage waveform (assuming 5HT waveform parameters)
        waveform_type = QSettings("HashemiLab", "NeuroStemVolt").value("waveform", "None", type=str)
        try:
            if waveform_type == "5HT":
                wf = Waveforms(0.2, [1.0, -0.1], 0.2, 1000, processed_data.shape[1])
            elif waveform_type == "DA":
                wf = Waveforms(-0.4, [1.3, -0.4], -0.4, 400, processed_data.shape[1])
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

        self._clear_figure()

        waveform_type = QSettings("HashemiLab", "NeuroStemVolt").value("waveform", "None", type=str)

        # Generate voltage waveform (assuming 5HT waveform parameters)
        try:
            if waveform_type == "5HT":
                wf = Waveforms(0.2, [1.0, -0.1], 0.2, 1000, processed_data.shape[1])
            elif waveform_type == "DA":
                wf = Waveforms(-0.4, [1.3, -0.4], -0.4, 400, processed_data.shape[1])
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


    def plot_it_time_series_single_experiment(self, group_analysis, experiment_index=0):
        """
        Plot consecutive IT traces from a single experiment with a color gradient.
        
        Each file's IT trace is plotted with increasing color intensity based on
        its position in the experimental timeline. Uses a colorbar instead of 
        individual legends for each timepoint.
        
        If calibration is enabled, plots concentration vs time instead of current vs time.
        
        Args:
            group_analysis: GroupAnalysis object containing experiments
            experiment_index (int): Index of the experiment to visualize
        """
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize as MplNormalize
        
        # Show loading dialog
        progress = QProgressDialog("Processing IT time series...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        experiments = group_analysis.get_experiments()
        if not experiments or experiment_index >= len(experiments):
            self.axes.clear()
            self.axes.set_title("No experiment data available")
            self.draw()
            progress.close()
            return
        
        experiment = experiments[experiment_index]
        n_files = experiment.get_file_count()
        
        if n_files == 0:
            self.axes.clear()
            self.axes.set_title("No files in experiment")
            self.draw()
            progress.close()
            return
        
        # Get settings
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)
        calibration_enabled = settings.value("calibration_enabled", False, type=bool)
        time_between_files = experiment.get_time_between_files()
        
        # Clear the figure and remove any existing colorbar
        self._clear_figure()
        
        # Create a colormap - using viridis which goes from purple/blue to yellow/green
        cmap = cm.viridis
        norm = MplNormalize(vmin=0, vmax=n_files - 1)
        
        # Plot each file's IT trace
        for file_idx in range(n_files):
            try:
                sf = experiment.get_spheroid_file(file_idx)
                it_trace = sf.get_processed_data_IT()
                
                if it_trace is None or len(it_trace) == 0:
                    continue
                
                # Create time axis in seconds
                time_sec = np.arange(len(it_trace)) / freq
                
                # Get color from colormap based on file index
                color = cmap(norm(file_idx))
                
                # Plot the trace
                self.axes.plot(time_sec, it_trace, color=color, linewidth=1.0, alpha=0.8)
                
            except Exception as e:
                print(f"Warning: Could not plot IT for file {file_idx}: {e}")
                continue
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Calculate time values for colorbar ticks
        time_values = np.arange(n_files) * time_between_files
        
        self.cbar = self.fig.colorbar(sm, ax=self.axes)
        self.cbar.set_label('Time in Experiment (min)', fontsize=10)
        
        # Set colorbar ticks to show time values
        # Tick positions should be in the data space (0 to n_files-1), not normalized
        n_ticks = min(6, n_files)  # Limit number of ticks
        tick_indices = np.linspace(0, n_files - 1, n_ticks, dtype=int)
        tick_labels = [f'{time_values[i]:.0f}' for i in tick_indices]
        self.cbar.set_ticks(tick_indices)  # Use actual file indices as tick positions
        self.cbar.set_ticklabels(tick_labels)
        
        # Axis labels
        self.axes.set_xlabel('Time (seconds)', fontsize=12)
        y_label = 'Concentration (nM)' if calibration_enabled else 'Current (nA)'
        self.axes.set_ylabel(y_label, fontsize=12)
        
        # Title - use folder name
        trace_type = 'C-T' if calibration_enabled else 'I-T'
        from ui.wizard_3_results.timepoint_dialog import get_experiment_display_name
        exp_name = get_experiment_display_name(experiment, experiment_index)
        title = f'{trace_type} Time Series - {exp_name}'
        self.axes.set_title(title, fontsize=14, fontweight='bold')
        
        self.axes.grid(True, alpha=0.3)
        
        # Set x-axis ticks
        if n_files > 0:
            try:
                sf = experiment.get_spheroid_file(0)
                it_trace = sf.get_processed_data_IT()
                max_t = len(it_trace) / freq if it_trace is not None else 60
                tick_interval = 5  # seconds
                ticks = np.arange(0, max_t + tick_interval, tick_interval)
                self.axes.set_xticks(ticks)
            except:
                pass
        
        self.fig.tight_layout()
        self.draw()
        progress.close()

    # LEGACY: This function has been deprecated in favor of plot_compare_amplitudes_grouped()
    # The UI button has been removed. Keeping for backward compatibility.
    # def plot_amplitudes_vs_timepoint_single_experiment(self, group_analysis, experiment_index=0):
    #     """
    #     Plot peak amplitudes across timepoints/files for a single stimulation experiment.
    #     
    #     Shows one amplitude value per file (extracted from peak detection), plotted 
    #     against file index or time. Each file collected at regular intervals 
    #     (e.g., every 10 min) shows one peak amplitude.
    #     
    #     If calibration is enabled, plots concentration instead of current.
    #     
    #     Args:
    #         group_analysis: GroupAnalysis object containing experiments
    #         experiment_index (int): Index of the experiment to visualize
    #     """
    #     # Show loading dialog
    #     progress = QProgressDialog("Processing amplitude data...", None, 0, 0, self)
    #     progress.setWindowModality(Qt.ApplicationModal)
    #     progress.setAutoClose(True)
    #     progress.setAutoReset(True)
    #     progress.setMinimumDuration(0)
    #     progress.show()
    #     QApplication.processEvents()
    #     
    #     experiments = group_analysis.get_experiments()
    #     if not experiments or experiment_index >= len(experiments):
    #         self.axes.clear()
    #         self.axes.set_title("No experiment data available")
    #         self.draw()
    #         progress.close()
    #         return
    #     
    #     experiment = experiments[experiment_index]
    #     n_files = experiment.get_file_count()
    #     
    #     if n_files == 0:
    #         self.axes.clear()
    #         self.axes.set_title("No files in experiment")
    #         self.draw()
    #         progress.close()
    #         return
    #     
    #     # Get settings
    #     settings = QSettings("HashemiLab", "NeuroStemVolt")
    #     calibration_enabled = settings.value("calibration_enabled", False, type=bool)
    #     time_between_files = experiment.get_time_between_files()
    #     files_before_treatment = experiment.get_number_of_files_before_treatment()
    #     
    #     # Collect amplitudes from each file
    #     timepoints = []
    #     amplitudes = []
    #     file_indices = []
    #     
    #     for file_idx in range(n_files):
    #         try:
    #             sf = experiment.get_spheroid_file(file_idx)
    #             metadata = sf.get_metadata()
    #             
    #             if metadata is None:
    #                 continue
    #             
    #             peak_amplitude_values = metadata.get('peak_amplitude_values', None)
    #             
    #             if peak_amplitude_values is None:
    #                 continue
    #             
    #             # Handle both single and multiple peak values
    #             if isinstance(peak_amplitude_values, (list, np.ndarray)):
    #                 if len(peak_amplitude_values) > 0:
    #                     # For stimulation files, typically take the main peak (first or max)
    #                     amp = float(np.max(np.abs(peak_amplitude_values)))
    #                 else:
    #                     continue
    #             else:
    #                 amp = float(peak_amplitude_values)
    #             
    #             # Calculate time point
    #             time_min = file_idx * time_between_files
    #             
    #             timepoints.append(time_min)
    #             amplitudes.append(amp)
    #             file_indices.append(file_idx)
    #             
    #         except Exception as e:
    #             print(f"Warning: Could not get amplitude for file {file_idx}: {e}")
    #             continue
    #     
    #     if not amplitudes:
    #         self.axes.clear()
    #         self.axes.set_title("No amplitude data available")
    #         self.draw()
    #         progress.close()
    #         return
    #     
    #     # Convert to arrays
    #     timepoints = np.array(timepoints)
    #     amplitudes = np.array(amplitudes)
    #     
    #     # Clear and plot
    #     self._clear_figure()
    #     
    #     # Plot amplitudes as connected line with markers
    #     self.axes.plot(timepoints, amplitudes, 'o-', color='#1f77b4', 
    #                   markersize=8, linewidth=2, label='Peak Amplitude')
    #     
    #     # Add treatment start line if applicable
    #     if files_before_treatment > 0:
    #         treatment_time = files_before_treatment * time_between_files
    #         self.axes.axvline(x=treatment_time, color='red', linestyle='--', 
    #                         linewidth=2, label='Treatment Start')
    #     
    #     # Axis labels
    #     self.axes.set_xlabel('Time (minutes)', fontsize=12)
    #     y_label = 'Peak Concentration (nM)' if calibration_enabled else 'Peak Amplitude (nA)'
    #     self.axes.set_ylabel(y_label, fontsize=12)
    #     
    #     # Title - use folder name
    #     value_type = 'Concentration' if calibration_enabled else 'Amplitude'
    #     from ui.wizard_3_results.timepoint_dialog import get_experiment_display_name
    #     exp_name = get_experiment_display_name(experiment, experiment_index)
    #     title = f'Peak {value_type} vs Timepoint - {exp_name}'
    #     self.axes.set_title(title, fontsize=14, fontweight='bold')
    #     
    #     self.axes.legend(fontsize=10, frameon=True, fancybox=True)
    #     self.axes.grid(True, alpha=0.3)
    #     
    #     # Set x-axis ticks at file collection intervals
    #     max_time = timepoints[-1] if len(timepoints) > 0 else 0
    #     tick_interval = time_between_files
    #     # Adjust tick interval if there are too many ticks
    #     if max_time / tick_interval > 15:
    #         tick_interval = max_time / 10
    #     ticks = np.arange(0, max_time + tick_interval, tick_interval)
    #     self.axes.set_xticks(ticks)
    #     
    #     # Add annotation showing number of files
    #     info_text = f'Files: {len(amplitudes)} | Interval: {time_between_files:.0f} min'
    #     self.axes.text(0.02, 0.98, info_text, transform=self.axes.transAxes,
    #                   fontsize=9, verticalalignment='top',
    #                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    #     
    #     self.fig.tight_layout()
    #     self.draw()
    #     progress.close()

    def plot_compare_its_at_timepoint(self, group_analysis, timepoint_idx, timepoint_label, selected_experiments):
        """
        Plot IT traces from multiple experiments at a specific timepoint with custom colors.
        
        Args:
            group_analysis: GroupAnalysis object containing experiments
            timepoint_idx (int): File index for the selected timepoint
            timepoint_label (str): Label for the timepoint (e.g., "-30 min")
            selected_experiments: List of (experiment_index, color_hex) tuples
        """
        # Show loading dialog
        progress = QProgressDialog("Processing IT comparison...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        experiments = group_analysis.get_experiments()
        if not experiments:
            self.axes.clear()
            self.axes.set_title("No experiment data available")
            self.draw()
            progress.close()
            return
        
        # Get settings
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)
        calibration_enabled = settings.value("calibration_enabled", False, type=bool)
        
        # Clear the figure and create new axes
        self._clear_figure()
        
        # Import helper for experiment names
        from ui.wizard_3_results.timepoint_dialog import get_experiment_display_name
        
        # Plot each selected experiment
        max_time_sec = 0
        for exp_idx, color in selected_experiments:
            if exp_idx >= len(experiments):
                continue
            
            experiment = experiments[exp_idx]
            
            # Check if the timepoint exists for this experiment
            if timepoint_idx >= experiment.get_file_count():
                continue
            
            try:
                sf = experiment.get_spheroid_file(timepoint_idx)
                it_trace = sf.get_processed_data_IT()
                
                if it_trace is None or len(it_trace) == 0:
                    continue
                
                # Create time axis in seconds
                time_sec = np.arange(len(it_trace)) / freq
                max_time_sec = max(max_time_sec, time_sec[-1])
                
                # Get experiment name for legend
                exp_name = get_experiment_display_name(experiment, exp_idx)
                
                # Plot the trace with custom color
                self.axes.plot(time_sec, it_trace, color=color, linewidth=2.5, 
                              alpha=0.9, label=exp_name)
                
            except Exception as e:
                print(f"Warning: Could not plot IT for experiment {exp_idx}: {e}")
                continue
        
        # Axis labels
        self.axes.set_xlabel('Time (seconds)', fontsize=12)
        y_label = 'Concentration (nM)' if calibration_enabled else 'Current (nA)'
        self.axes.set_ylabel(y_label, fontsize=12)
        
        # Title
        trace_type = 'C-T' if calibration_enabled else 'I-T'
        self.axes.set_title(f'{trace_type} Comparison at {timepoint_label}', 
                           fontsize=14, fontweight='bold')
        
        self.axes.legend(fontsize=9, frameon=True, fancybox=True, loc='best')
        self.axes.grid(True, alpha=0.3)
        
        # Set x-axis ticks
        if max_time_sec > 0:
            tick_interval = 5  # seconds
            ticks = np.arange(0, max_time_sec + tick_interval, tick_interval)
            self.axes.set_xticks(ticks)
        
        self.fig.tight_layout()
        self.draw()
        progress.close()

    def plot_compare_amplitudes_multi_exp(self, group_analysis, selected_experiments):
        """
        Plot amplitude timelines from multiple experiments with custom colors.
        
        Args:
            group_analysis: GroupAnalysis object containing experiments
            selected_experiments: List of (experiment_index, color_hex) tuples
        """
        # Show loading dialog
        progress = QProgressDialog("Processing amplitude comparison...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        experiments = group_analysis.get_experiments()
        if not experiments:
            self.axes.clear()
            self.axes.set_title("No experiment data available")
            self.draw()
            progress.close()
            return
        
        # Get settings
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        calibration_enabled = settings.value("calibration_enabled", False, type=bool)
        
        
        # Clear the figure and create new axes
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        
        # Create a colormap - using viridis which goes from purple/blue to yellow/green
        cmap = cm.viridis
        norm = MplNormalize(vmin=0, vmax=n_files - 1)
        
        # Plot each file's IT trace
        for file_idx in range(n_files):
            try:
                sf = experiment.get_spheroid_file(file_idx)
                it_trace = sf.get_processed_data_IT()
                
                if it_trace is None or len(it_trace) == 0:
                    continue
                
                # Create time axis in seconds
                time_sec = np.arange(len(it_trace)) / freq
                
                # Get color from colormap based on file index
                color = cmap(norm(file_idx))
                
                # Plot the trace
                self.axes.plot(time_sec, it_trace, color=color, linewidth=1.0, alpha=0.8)
                
            except Exception as e:
                print(f"Warning: Could not plot IT for file {file_idx}: {e}")
                continue
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Calculate time values for colorbar ticks
        time_values = np.arange(n_files) * time_between_files
        
        self.cbar = self.fig.colorbar(sm, ax=self.axes)
        self.cbar.set_label('Time in Experiment (min)', fontsize=10)
        
        # Set colorbar ticks to show time values
        # Tick positions should be in the data space (0 to n_files-1), not normalized
        n_ticks = min(6, n_files)  # Limit number of ticks
        tick_indices = np.linspace(0, n_files - 1, n_ticks, dtype=int)
        tick_labels = [f'{time_values[i]:.0f}' for i in tick_indices]
        self.cbar.set_ticks(tick_indices)  # Use actual file indices as tick positions
        self.cbar.set_ticklabels(tick_labels)
        
        # Axis labels
        self.axes.set_xlabel('Time (minutes)', fontsize=12)
        y_label = 'Peak Concentration (nM)' if calibration_enabled else 'Peak Amplitude (nA)'
        self.axes.set_ylabel(y_label, fontsize=12)
        
        
        # Title
        trace_type = 'C-T' if calibration_enabled else 'I-T'
        from ui.wizard_3_results.timepoint_dialog import get_experiment_display_name
        exp_name = get_experiment_display_name(experiment, experiment_index)
        title = f'{trace_type} Time Series - {exp_name}'
        self.axes.set_title(title, fontsize=14, fontweight='bold')
        
        self.axes.grid(True, alpha=0.3)
        
        # Set x-axis ticks
        if max_time > 0:
            tick_interval = time_between_files
            if max_time / tick_interval > 15:
                tick_interval = max_time / 10
            ticks = np.arange(0, max_time + tick_interval, tick_interval)
            self.axes.set_xticks(ticks)
        
        # Add annotation showing number of experiments
        info_text = f'Experiments: {len(selected_experiments)}'
        self.axes.text(0.02, 0.98, info_text, transform=self.axes.transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.fig.tight_layout()
        self.draw()
        progress.close()

    
    def plot_amplitudes_vs_timepoint_single_experiment_legacy(self, group_analysis, experiment_index=0):
        """
        LEGACY: Plot peak amplitudes across timepoints/files for a single stimulation experiment.
        
        This function is kept for backward compatibility but the UI button has been removed.
        Use plot_compare_amplitudes_grouped() for the new grouped comparison functionality.
        
        Args:
            group_analysis: GroupAnalysis object containing experiments
            experiment_index (int): Index of the experiment to visualize
        """
        # Show loading dialog
        progress = QProgressDialog("Processing amplitude data...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        experiments = group_analysis.get_experiments()
        if not experiments or experiment_index >= len(experiments):
            self.axes.clear()
            self.axes.set_title("No experiment data available")
            self.draw()
            progress.close()
            return
        
        experiment = experiments[experiment_index]
        n_files = experiment.get_file_count()
        
        if n_files == 0:
            self.axes.clear()
            self.axes.set_title("No files in experiment")
            self.draw()
            progress.close()
            return
        
        # Get settings
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        calibration_enabled = settings.value("calibration_enabled", False, type=bool)
        time_between_files = experiment.get_time_between_files()
        files_before_treatment = experiment.get_number_of_files_before_treatment()
        
        # Collect amplitudes from each file
        timepoints = []
        amplitudes = []
        file_indices = []
        
        for file_idx in range(n_files):
            try:
                sf = experiment.get_spheroid_file(file_idx)
                metadata = sf.get_metadata()
                
                if metadata is None:
                    continue
                
                peak_amplitude_values = metadata.get('peak_amplitude_values', None)
                
                if peak_amplitude_values is None:
                    continue
                
                # Handle both single and multiple peak values
                if isinstance(peak_amplitude_values, (list, np.ndarray)):
                    if len(peak_amplitude_values) > 0:
                        amp = float(np.max(np.abs(peak_amplitude_values)))
                    else:
                        continue
                else:
                    amp = float(peak_amplitude_values)
                
                # Calculate time point
                time_min = file_idx * time_between_files
                
                timepoints.append(time_min)
                amplitudes.append(amp)
                file_indices.append(file_idx)
                
            except Exception as e:
                print(f"Warning: Could not get amplitude for file {file_idx}: {e}")
                continue
        
        if not amplitudes:
            self.axes.clear()
            self.axes.set_title("No amplitude data available")
            self.draw()
            progress.close()
            return
        
        # Convert to arrays
        timepoints = np.array(timepoints)
        amplitudes = np.array(amplitudes)
        
        # Clear and plot
        self._clear_figure()
        
        # Plot amplitudes as connected line with markers
        self.axes.plot(timepoints, amplitudes, 'o-', color='#1f77b4', 
                      markersize=8, linewidth=2, label='Peak Amplitude')
        
        # Add treatment start line if applicable
        if files_before_treatment > 0:
            treatment_time = files_before_treatment * time_between_files
            self.axes.axvline(x=treatment_time, color='red', linestyle='--', 
                            linewidth=2, label='Treatment Start')
        
        # Axis labels
        self.axes.set_xlabel('Time (minutes)', fontsize=12)
        y_label = 'Peak Concentration (nM)' if calibration_enabled else 'Peak Amplitude (nA)'
        self.axes.set_ylabel(y_label, fontsize=12)
        
        # Title - use folder name
        value_type = 'Concentration' if calibration_enabled else 'Amplitude'
        from ui.wizard_3_results.timepoint_dialog import get_experiment_display_name
        exp_name = get_experiment_display_name(experiment, experiment_index)
        title = f'[LEGACY] Peak {value_type} vs Timepoint - {exp_name}'
        self.axes.set_title(title, fontsize=14, fontweight='bold')
        
        self.axes.legend(fontsize=10, frameon=True, fancybox=True)
        self.axes.grid(True, alpha=0.3)
        
        # Set x-axis ticks at file collection intervals
        max_time = timepoints[-1] if len(timepoints) > 0 else 0
        tick_interval = time_between_files
        if max_time / tick_interval > 15:
            tick_interval = max_time / 10
        ticks = np.arange(0, max_time + tick_interval, tick_interval)
        self.axes.set_xticks(ticks)
        
        # Add annotation showing number of files
        info_text = f'Files: {len(amplitudes)} | Interval: {time_between_files:.0f} min'
        self.axes.text(0.02, 0.98, info_text, transform=self.axes.transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.fig.tight_layout()
        self.draw()
        progress.close()

    def plot_compare_tau_grouped(self, group_analysis, timepoint_idx, timepoint_label, groups):
        """
        Compare tau values from grouped experiments at a specific timepoint.
        
        Shows a scatter plot with mean markers and error bars, where each group
        is displayed on the x-axis. Experiments in the same group are averaged.
        
        Args:
            group_analysis: GroupAnalysis object containing experiments
            timepoint_idx (int): File index for the selected timepoint
            timepoint_label (str): Label for the timepoint (e.g., "-30 min")
            groups: List of dicts with 'name', 'color', 'experiments' keys
        """
        from scipy.optimize import curve_fit
        
        # Show loading dialog
        progress = QProgressDialog("Processing tau comparison...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        experiments = group_analysis.get_experiments()
        if not experiments:
            self.axes.clear()
            self.axes.set_title("No experiment data available")
            self.draw()
            progress.close()
            return
        
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)
        
        # Exponential decay function
        def exp_decay(t, A, k, C):
            return (A - C) * np.exp(-k * t) + C
        
        # Clear the figure
        self._clear_figure()
        
        # Collect tau values for each group
        group_names = []
        group_colors = []
        all_taus = []  # List of lists - tau values for each group
        group_means = []
        group_stds = []
        
        for group in groups:
            group_name = group['name']
            group_color = group['color']
            exp_indices = group['experiments']
            
            taus_in_group = []
            
            for exp_idx in exp_indices:
                if exp_idx >= len(experiments):
                    continue
                
                experiment = experiments[exp_idx]
                
                # Check if the timepoint exists
                if timepoint_idx >= experiment.get_file_count():
                    continue
                
                try:
                    sf = experiment.get_spheroid_file(timepoint_idx)
                    it_trace = sf.get_processed_data_IT()
                    metadata = sf.get_metadata()
                    
                    if it_trace is None or len(it_trace) == 0:
                        continue
                    
                    # Get peak position from metadata
                    peak_pos = None
                    if metadata and 'peak_amplitude_positions' in metadata:
                        peak_indices = metadata['peak_amplitude_positions']
                        if isinstance(peak_indices, (list, np.ndarray)) and len(peak_indices) > 0:
                            peak_pos = int(peak_indices[0])
                        elif isinstance(peak_indices, (int, np.integer)):
                            peak_pos = int(peak_indices)
                    
                    if peak_pos is None:
                        # Find peak as max value
                        peak_pos = int(np.argmax(it_trace))
                    
                    # Extract post-peak decay
                    decay_samples = min(400, len(it_trace) - peak_pos - 1)
                    if decay_samples < 20:
                        continue
                    
                    post_peak = it_trace[peak_pos:peak_pos + decay_samples]
                    t_decay = np.arange(len(post_peak)) / freq
                    
                    # Fit exponential decay
                    try:
                        A0 = post_peak[0]
                        C0 = post_peak[-1]
                        k0 = 0.1
                        
                        popt, _ = curve_fit(exp_decay, t_decay, post_peak, 
                                           p0=[A0, k0, C0],
                                           bounds=([0, 0.001, -np.inf], [np.inf, 10, np.inf]),
                                           maxfev=5000)
                        
                        A_fit, k_fit, C_fit = popt
                        tau = 1.0 / k_fit if k_fit > 0 else np.nan
                        
                        if not np.isnan(tau) and tau > 0 and tau < 100:  # Sanity check
                            taus_in_group.append(tau)
                    except:
                        continue
                        
                except Exception as e:
                    print(f"Warning: Could not process experiment {exp_idx}: {e}")
                    continue
            
            if taus_in_group:
                group_names.append(group_name)
                group_colors.append(group_color)
                all_taus.append(taus_in_group)
                group_means.append(np.mean(taus_in_group))
                group_stds.append(np.std(taus_in_group))
        
        if not group_names:
            self.axes.clear()
            self.axes.set_title("No valid tau data found")
            self.draw()
            progress.close()
            return
        
        # Calculate y-axis limits with padding
        all_values = group_means
        all_stds_vals = group_stds
        y_min = min([v - 2*s for v, s in zip(all_values, all_stds_vals)])
        y_max = max([v + 2*s for v, s in zip(all_values, all_stds_vals)])
        y_max = y_max * 1.15  # Add padding for annotations
        y_min = max(0, y_min * 0.9)  # Keep y_min >= 0 for tau
        
        self.axes.set_ylim(bottom=y_min, top=y_max)
        
        # Create scatter plot with horizontal lines and error bars
        x_positions = np.arange(len(group_names))
        
        for i, (x, mean_val, std_val, color, taus) in enumerate(zip(
            x_positions, group_means, group_stds, group_colors, all_taus)):
            
            # Scatter point for mean
            self.axes.scatter([x], [mean_val], s=100, alpha=0.7, color=color, zorder=3)
            
            # Horizontal line at mean
            self.axes.hlines(mean_val, xmin=x-0.2, xmax=x+0.2, colors=color, 
                           linewidth=3, alpha=0.5)
            
            # Error bars
            self.axes.errorbar(x, mean_val, yerr=std_val, fmt='none', 
                             ecolor=color, capsize=5, capthick=2, linewidth=2, alpha=0.5)
            
            # Individual data points with jitter
            jitter = (np.random.random(len(taus)) - 0.5) * 0.15
            self.axes.scatter(x + jitter, taus, color=color, s=30, 
                            alpha=0.4, edgecolor='none', zorder=2)
        
        # Axis labels and title
        self.axes.set_xticks(x_positions)
        self.axes.set_xticklabels(group_names, fontsize=12)
        self.axes.set_ylabel('Tau (seconds)', fontsize=12)
        self.axes.set_xlabel('Treatment', fontsize=14)
        self.axes.set_title(f'Tau Comparison at {timepoint_label}', fontsize=14, fontweight='bold')
        self.axes.grid(axis='y', alpha=0.3)
        
        # Add sample size annotation above each group
        for i, (x, mean_val, std_val, taus) in enumerate(zip(x_positions, group_means, group_stds, all_taus)):
            y_pos = mean_val + std_val + (y_max - y_min) * 0.05
            self.axes.annotate(f'n={len(taus)}', (x, y_pos), ha='center', fontsize=9)
        
        self.fig.tight_layout()
        self.draw()
        progress.close()

    def plot_compare_amplitudes_grouped(self, group_analysis, timepoint_idx, timepoint_label, groups):
        """
        Compare amplitudes from grouped experiments at a specific timepoint.
        
        Shows a scatter plot with mean markers and error bars, where each group
        is displayed on the x-axis. Experiments in the same group are averaged.
        
        Args:
            group_analysis: GroupAnalysis object containing experiments
            timepoint_idx (int): File index for the selected timepoint
            timepoint_label (str): Label for the timepoint (e.g., "-30 min")
            groups: List of dicts with 'name', 'color', 'experiments' keys
        """
        # Show loading dialog
        progress = QProgressDialog("Processing amplitude comparison...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        experiments = group_analysis.get_experiments()
        if not experiments:
            self.axes.clear()
            self.axes.set_title("No experiment data available")
            self.draw()
            progress.close()
            return
        
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        calibration_enabled = settings.value("calibration_enabled", False, type=bool)
        
        # Clear the figure
        self._clear_figure()
        
        # Collect amplitude values for each group
        group_names = []
        group_colors = []
        all_amps = []  # List of lists - amplitude values for each group
        group_means = []
        group_stds = []
        
        for group in groups:
            group_name = group['name']
            group_color = group['color']
            exp_indices = group['experiments']
            
            amps_in_group = []
            
            for exp_idx in exp_indices:
                if exp_idx >= len(experiments):
                    continue
                
                experiment = experiments[exp_idx]
                
                if timepoint_idx >= experiment.get_file_count():
                    continue
                
                try:
                    sf = experiment.get_spheroid_file(timepoint_idx)
                    metadata = sf.get_metadata()
                    
                    if metadata is None:
                        continue
                    
                    peak_amplitude_values = metadata.get('peak_amplitude_values', None)
                    
                    if peak_amplitude_values is None:
                        continue
                    
                    # Handle both single and multiple peak values
                    if isinstance(peak_amplitude_values, (list, np.ndarray)):
                        if len(peak_amplitude_values) > 0:
                            amp = float(np.max(np.abs(peak_amplitude_values)))
                        else:
                            continue
                    else:
                        amp = float(peak_amplitude_values)
                    
                    amps_in_group.append(amp)
                    
                except Exception as e:
                    print(f"Warning: Could not get amplitude for exp {exp_idx}: {e}")
                    continue
            
            if amps_in_group:
                group_names.append(group_name)
                group_colors.append(group_color)
                all_amps.append(amps_in_group)
                group_means.append(np.mean(amps_in_group))
                group_stds.append(np.std(amps_in_group))
        
        if not group_names:
            self.axes.clear()
            self.axes.set_title("No valid amplitude data found")
            self.draw()
            progress.close()
            return
        
        # Calculate y-axis limits with padding
        all_values = group_means
        all_stds_vals = group_stds
        y_min = min([v - 2*s for v, s in zip(all_values, all_stds_vals)])
        y_max = max([v + 2*s for v, s in zip(all_values, all_stds_vals)])
        y_max = y_max * 1.15  # Add padding for annotations
        y_min = max(0, y_min * 0.9)  # Keep y_min >= 0 for amplitude
        
        self.axes.set_ylim(bottom=y_min, top=y_max)
        
        # Create scatter plot with horizontal lines and error bars
        x_positions = np.arange(len(group_names))
        
        for i, (x, mean_val, std_val, color, amps) in enumerate(zip(
            x_positions, group_means, group_stds, group_colors, all_amps)):
            
            # Scatter point for mean
            self.axes.scatter([x], [mean_val], s=100, alpha=0.7, color=color, zorder=3)
            
            # Horizontal line at mean
            self.axes.hlines(mean_val, xmin=x-0.2, xmax=x+0.2, colors=color, 
                           linewidth=3, alpha=0.5)
            
            # Error bars
            self.axes.errorbar(x, mean_val, yerr=std_val, fmt='none', 
                             ecolor=color, capsize=5, capthick=2, linewidth=2, alpha=0.5)
            
            # Individual data points with jitter
            jitter = (np.random.random(len(amps)) - 0.5) * 0.15
            self.axes.scatter(x + jitter, amps, color=color, s=30, 
                            alpha=0.4, edgecolor='none', zorder=2)
        
        # Axis labels and title
        self.axes.set_xticks(x_positions)
        self.axes.set_xticklabels(group_names, fontsize=12)
        
        y_label = 'Peak Concentration (nM)' if calibration_enabled else 'Peak Amplitude (nA)'
        self.axes.set_ylabel(y_label, fontsize=12)
        self.axes.set_xlabel('Treatment', fontsize=14)
        
        value_type = 'Concentration' if calibration_enabled else 'Amplitude'
        self.axes.set_title(f'{value_type} Comparison at {timepoint_label}', fontsize=14, fontweight='bold')
        self.axes.grid(axis='y', alpha=0.3)
        
        # Add sample size annotation above each group
        for i, (x, mean_val, std_val, amps) in enumerate(zip(x_positions, group_means, group_stds, all_amps)):
            y_pos = mean_val + std_val + (y_max - y_min) * 0.05
            self.axes.annotate(f'n={len(amps)}', (x, y_pos), ha='center', fontsize=9)
        
        self.fig.tight_layout()
        self.draw()
        progress.close()

    def plot_amplitude_violin(self, group_analysis, timepoints, groups, plot_title, ylabel):
        """
        Create a violin/swarm plot comparing amplitude distributions between groups.
        
        Uses seaborn for the violin and swarm plot visualization.
        
        Args:
            group_analysis: GroupAnalysis object containing experiments
            timepoints: List of [(idx1, label1), (idx2, label2)] - exactly two timepoints
            groups: List of dicts with 'name', 'color', 'experiments' keys
            plot_title (str): Custom title for the plot
            ylabel (str): Custom y-axis label
        """
        try:
            import seaborn as sns
            import pandas as pd
        except ImportError:
            self.axes.clear()
            self.axes.set_title("Seaborn library required for violin plots")
            self.draw()
            return
        
        # Show loading dialog
        progress = QProgressDialog("Creating violin plot...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        experiments = group_analysis.get_experiments()
        if not experiments:
            self.axes.clear()
            self.axes.set_title("No experiment data available")
            self.draw()
            progress.close()
            return
        
        # Collect data into a DataFrame format
        data_rows = []
        palette = {}
        
        for group in groups:
            group_name = group['name']
            group_color = group['color']
            exp_indices = group['experiments']
            
            palette[group_name] = group_color
            
            for exp_idx in exp_indices:
                if exp_idx >= len(experiments):
                    continue
                
                experiment = experiments[exp_idx]
                
                # Get amplitudes for the selected timepoints
                for tp_idx, tp_label in timepoints:
                    if tp_idx >= experiment.get_file_count():
                        continue
                    
                    try:
                        sf = experiment.get_spheroid_file(tp_idx)
                        metadata = sf.get_metadata()
                        
                        if metadata is None:
                            continue
                        
                        peak_amplitude_values = metadata.get('peak_amplitude_values', None)
                        
                        if peak_amplitude_values is None:
                            continue
                        
                        # Handle both single and multiple peak values
                        if isinstance(peak_amplitude_values, (list, np.ndarray)):
                            if len(peak_amplitude_values) > 0:
                                amp = float(np.max(np.abs(peak_amplitude_values)))
                            else:
                                continue
                        else:
                            amp = float(peak_amplitude_values)
                        
                        data_rows.append({
                            'Group': group_name,
                            'Amplitude': amp,
                            'Timepoint': tp_label,
                            'Experiment': exp_idx
                        })
                        
                    except Exception as e:
                        print(f"Warning: Could not get amplitude for exp {exp_idx}: {e}")
                        continue
        
        if not data_rows:
            self.axes.clear()
            self.axes.set_title("No amplitude data available")
            self.draw()
            progress.close()
            return
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Clear the figure
        self._clear_figure()
        
        # Create violin plot with swarm overlay
        try:
            # Violin plot as base
            sns.violinplot(
                data=df,
                y='Amplitude',
                x='Group',
                hue='Group',
                palette=palette,
                alpha=0.3,
                inner=None,
                linewidth=0,
                legend=False,
                ax=self.axes
            )
            
            # Overlay swarm plot with colored points
            sns.swarmplot(
                data=df,
                y='Amplitude',
                x='Group',
                hue='Group',
                palette=palette,
                alpha=0.6,
                size=7,
                legend=False,
                ax=self.axes
            )
            
        except Exception as e:
            print(f"Warning: Swarm/violin plot failed, using simpler plot: {e}")
            # Fallback to boxplot if seaborn fails
            self.axes.clear()
            group_names = list(palette.keys())
            for i, gname in enumerate(group_names):
                gdata = df[df['Group'] == gname]['Amplitude'].values
                bp = self.axes.boxplot([gdata], positions=[i], widths=0.6, 
                                       patch_artist=True)
                bp['boxes'][0].set_facecolor(palette[gname])
                bp['boxes'][0].set_alpha(0.5)
            self.axes.set_xticks(range(len(group_names)))
            self.axes.set_xticklabels(group_names)
        
        # Styling
        self.axes.set_xlabel("Group", fontsize=12)
        self.axes.set_ylabel(ylabel, fontsize=12)
        self.axes.set_title(plot_title, fontsize=13, fontweight='bold')
        
        # Create custom legend
        import matplotlib.pyplot as plt
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=8, label=name)
            for name, color in palette.items()
        ]
        self.axes.legend(handles=legend_handles, loc='best', fontsize=9)
        
        self.axes.grid(alpha=0.3)
        
        # Add sample size annotations
        group_names = df['Group'].unique()
        for i, gname in enumerate(group_names):
            n = len(df[df['Group'] == gname])
            y_max = df[df['Group'] == gname]['Amplitude'].max()
            self.axes.annotate(f'n={n}', (i, y_max), 
                             xytext=(0, 10), textcoords='offset points',
                             ha='center', fontsize=9)
        
        self.fig.tight_layout()
        self.draw()
        progress.close()