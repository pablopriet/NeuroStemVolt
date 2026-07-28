from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QDialog, QInputDialog,
    QMessageBox, QProgressDialog, QApplication
)
from PyQt5.QtCore import QSettings, Qt

from core.output_manager import OutputManager
from core.processing import *

from ui.utils.styles import apply_custom_styles
from ui.widgets.plot_canvas import PlotCanvas
from ui.wizard_3_results.timepoint_dialog import (
    TimepointSelectionDialog, TimepointMultiSelectionDialog)
import os

### Third Page

class ResultsPage(QWizardPage):
    """
    Wizard page for visualizing and exporting group-level analysis results.

    Provides options to:
    - Plot amplitude metrics over time.
    - Visualize decay fitting and tau parameters.
    - Export metrics and figures to files.

    Attributes:
        result_plot (PlotCanvas): Matplotlib canvas for displaying results.
        analysis_buttons (list): List of QPushButtons tied to specific analysis/export actions.
        placeholder (QLabel): Placeholder shown before any plot is displayed.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        def _sec(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "font-weight: bold; font-size: 9pt;"
                " padding-top: 6px; border-top: 1px solid palette(mid);")
            return lbl

        # ── Analysis buttons ─────────────────────────────────────────

        # Available for both
        btn_avg = QPushButton("Mean Amplitude Over Experiments"); apply_custom_styles(btn_avg)
        btn_amp = QPushButton("Individual Amplitudes Per Replicate"); apply_custom_styles(btn_amp)
        btn_mean_its = QPushButton("Mean ITs"); apply_custom_styles(btn_mean_its)

        # Recommended for Single Peak
        btn_fit   = QPushButton("Exponential Decay Fitting"); apply_custom_styles(btn_fit)
        btn_param = QPushButton("Tau-Time");             apply_custom_styles(btn_param)

        # Recommended for Multi-Peak
        self.btn_spont_freq     = QPushButton("Peak Frequency");            apply_custom_styles(self.btn_spont_freq)
        self.btn_spont_windowed = QPushButton("Amplitudes per Time Window"); apply_custom_styles(self.btn_spont_windowed)
        self.spontaneous_buttons = [self.btn_spont_freq, self.btn_spont_windowed]

        for btn in self.spontaneous_buttons:
            btn.hide()

        # Export buttons
        btn_save     = QPushButton("Save Current Plot");     apply_custom_styles(btn_save)
        btn_save_all = QPushButton("Save All Plots");        apply_custom_styles(btn_save_all)
        btn_export   = QPushButton("Export Metrics as CSV"); apply_custom_styles(btn_export)

        self.analysis_buttons = [btn_avg, btn_fit, btn_param, btn_amp, btn_mean_its, btn_save, btn_save_all, btn_export]

        # ── Connect buttons ──────────────────────────────────────────
        for btn, fn in (
            (btn_avg,   lambda: self.result_plot.show_average_over_experiments(self.wizard().group_analysis)),
            (btn_fit,   self.handle_decay_fit),
            (btn_param, lambda: self.result_plot.show_tau_param_over_time(self.wizard().group_analysis)),
            (btn_amp,   lambda: self.result_plot.show_amplitudes_over_time(self.wizard().group_analysis)),
            (btn_mean_its, lambda: self.result_plot.show_mean_ITs(self.wizard().group_analysis)),
            (self.btn_spont_freq,     lambda: self.result_plot.show_spontaneous_peak_frequency(self.wizard().group_analysis)),
            (self.btn_spont_windowed, self.handle_windowed_amplitudes),
        ):
            btn.clicked.connect(lambda _, f=fn: self._reveal_and_call(f))

        btn_save.clicked.connect(self.save_current_plot)
        btn_save_all.clicked.connect(self.save_all_plots)
        btn_export.clicked.connect(self.export_all_as_csv)

        # ── Left panel ───────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(4)

        left.addWidget(_sec("Available for All"))
        left.addWidget(btn_avg)
        left.addWidget(btn_amp)
        left.addWidget(btn_mean_its)

        left.addWidget(_sec("Single Peak"))
        left.addWidget(btn_param)
        left.addWidget(btn_fit)

        self.lbl_multi_peak = _sec("Multi-Peak")
        self.lbl_multi_peak.hide()
        left.addWidget(self.lbl_multi_peak)
        left.addWidget(self.btn_spont_freq)
        left.addWidget(self.btn_spont_windowed)

        left.addStretch()

        left.addWidget(_sec("Export"))
        left.addWidget(btn_save)
        left.addWidget(btn_save_all)
        left.addWidget(btn_export)

        # ── Right panel (plot) ───────────────────────────────────────
        self.placeholder = QLabel("Select an analysis option to show plot")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: palette(windowtext); font-size: 11pt;")
        self.result_plot = PlotCanvas(self, width=6, height=5)
        self.result_plot.hide()

        right = QVBoxLayout()
        right.addWidget(self.placeholder, stretch=1)
        right.addWidget(self.result_plot, stretch=1)

        # ── Main layout ──────────────────────────────────────────────
        content = QHBoxLayout()
        content.addLayout(left, 1)
        content.addLayout(right, 3)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(content, stretch=1)

        footer = QLabel("© 2026 Hashemi Lab · NeuroStemVolt · v1.1.1")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: palette(windowtext); font-size: 10pt;")
        main_layout.addWidget(footer)

        self.setLayout(main_layout)

    # ── Windowed amplitude plot ──────────────────────────────────────────────

    AMPLITUDE_WINDOW_KEY = "amplitude_window_sec"

    def _amplitude_window_sec(self):
        """Return the last window width the user chose (default 60 s)."""
        return QSettings("HashemiLab", "NeuroStemVolt").value(
            self.AMPLITUDE_WINDOW_KEY, 60.0, type=float)

    def handle_windowed_amplitudes(self):
        """
        Ask for an averaging window, then plot mean peak amplitude per window.

        Peaks are placed on the experiment's absolute time axis, so a window
        collects every peak recorded within that span regardless of which file
        it came from.
        """
        window, ok = QInputDialog.getDouble(
            self, "Averaging Window",
            "Average peak amplitudes over windows of (seconds):",
            self._amplitude_window_sec(), 1.0, 100000.0, 1)
        if not ok:
            return
        QSettings("HashemiLab", "NeuroStemVolt").setValue(self.AMPLITUDE_WINDOW_KEY, window)
        self.result_plot.show_windowed_peak_amplitudes(
            self.wizard().group_analysis, window_sec=window)

    def _save_canvas(self, output_folder, base_name):
        """
        Write the current canvas to the output folder as both PNG and SVG.

        The SVG mirrors the colorplot wizard's exports so individual elements
        stay editable in vector software.

        Args:
            output_folder (str): Destination directory.
            base_name (str): File name without extension.
        """
        plots_folder = os.path.join(output_folder, "plots")
        os.makedirs(plots_folder, exist_ok=True)
        self.result_plot.fig.savefig(
            os.path.join(plots_folder, f"{base_name}.png"), dpi=300, bbox_inches='tight')
        self.result_plot.fig.savefig(
            os.path.join(plots_folder, f"{base_name}.svg"), bbox_inches='tight')

    def _reveal_and_call(self, plot_fn):
        """
        Reveal the plot canvas and call a plotting function.

        Args:
            plot_fn (callable): A function that renders to `self.result_plot`.
        """
        if self.placeholder.isVisible():
            self.placeholder.hide()
            self.result_plot.show()
        plot_fn()

    def export_all_as_csv(self):
        """
        Export all computed group-level metrics as CSV files.

        Metrics include:
            - I-T traces
            - Amplitudes
            - Reuptake curves
            - Exponential fit parameters
            - AUC values

        Displays status dialogs and error messages as needed.
        """
        output_folder = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        if not output_folder or not os.path.isdir(output_folder):
            QMessageBox.warning(
                self, "No Output Folder",
                "Please set a valid output folder in Experiment Settings."
            )
            return

        # create & show the indeterminate progress dialog
        progress = QProgressDialog("Exporting CSV files…", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)    # hide the cancel button
        progress.setMinimumDuration(0)    # show immediately
        progress.setAutoClose(False)
        progress.show()
        QApplication.processEvents()      # force a repaint

        try:
            ga = self.wizard().group_analysis
            OutputManager.save_all_ITs(ga, output_folder)
            OutputManager.save_all_peak_amplitudes(ga, output_folder)
            OutputManager.save_all_reuptake_curves(ga, output_folder)
            # Exponential fitting: the JOINT method is the only one still in use.
            # The per-replicate-then-pooled, global/legacy, shared-k and two-stage
            # savers are commented out in OutputManager.
            # OutputManager.save_all_exponential_fitting_params(ga, output_folder)
            # OutputManager.save_all_exponential_fitting_params_global(ga, output_folder)
            # Guarded on its own so a fit failure cannot abort the rest of the export.
            try:
                OutputManager.save_exp_fit_joint(ga, output_folder)
            except Exception as ex:
                print(f"[export] save_exp_fit_joint failed: {ex}")

            OutputManager.save_all_AUC(ga, output_folder)

            # Export spontaneous peak metrics if applicable
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            file_type = settings.value("file_type", "None", type=str)
            if file_type == "Multi-Peak":
                OutputManager.save_spontaneous_peak_metrics(ga, output_folder)

            # Always generate experiment log for reproducibility
            OutputManager.save_experiment_log(ga, output_folder, qsettings=settings)

            # Auto-save peak session so the analysis can be reopened later
            colorplot_page = self.wizard().page(1)
            if hasattr(colorplot_page, "export_session_to_path"):
                session_path = os.path.join(output_folder, "peak_session.nsv")
                colorplot_page.export_session_to_path(session_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Export Failed",
                f"An error occurred while exporting:\n{e}"
            )
        else:
            QMessageBox.information(
                self, "CSV Export Complete",
                f"All metrics exported to:\n{output_folder}"
            )
        finally:
            progress.hide()

    def save_current_plot(self):
        """
        Save the currently displayed result plot as a PNG image.
        
        Prompts the user for a filename using QFileDialog, defaulting to the
        output folder's "plots" subfolder.
        """
        output_folder = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        default_dir = ""
        if output_folder and os.path.isdir(output_folder):
            default_dir = os.path.join(output_folder, "plots")
            os.makedirs(default_dir, exist_ok=True)

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current Plot As",
            default_dir,
            "PNG Files (*.png);;SVG Files (*.svg);;All Files (*)",
            options=options
        )
        if file_path:
            self.result_plot.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            svg_path = os.path.splitext(file_path)[0] + ".svg"
            if svg_path != file_path:
                self.result_plot.fig.savefig(svg_path, bbox_inches='tight')
            QMessageBox.information(self, "Plot Saved", f"Plot saved to:\n{file_path}")

    def save_all_plots(self):
        """
        Save all result plots to the configured output directory.

        Plots include:
            - Mean I-T traces
            - Tau parameter over time
            - Exponential decay fits
            - Amplitude trajectories

        Displays a progress dialog during export and handles exceptions.
        """
        output_folder = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        group_analysis = self.wizard().group_analysis
        if not output_folder or not os.path.isdir(output_folder):
            QMessageBox.warning(self, "No Output Folder", "Please set a valid output folder in Experiment Settings.")
            return

        # Ask which timepoints get an exponential-fit plot. This used to be
        # hard-coded to the first timepoint, so only one fit was ever exported.
        fit_indices = self._ask_exponential_fit_timepoints(group_analysis)
        if fit_indices is None:
            return  # user cancelled

        # create & show the indeterminate progress dialog
        progress = QProgressDialog("Exporting plots...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)    # hide the cancel button
        progress.setMinimumDuration(0)    # show immediately
        progress.setAutoClose(False)
        progress.show()
        QApplication.processEvents()      # force a repaint

        fit_failures = []
        try:
            # Save all group-level plots using OutputManager
            OutputManager.save_mean_ITs_plot(group_analysis, output_folder)
            OutputManager.save_plot_tau_over_time(group_analysis, output_folder)
            _saved, fit_failures = OutputManager.save_plots_exponential_fit_timepoints(
                group_analysis, output_folder, fit_indices)
            OutputManager.save_plot_all_amplitudes_over_time(group_analysis, output_folder)
            OutputManager.save_plot_mean_amplitudes_over_time(group_analysis, output_folder)

            # Save spontaneous analysis plots if applicable
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            file_type = settings.value("file_type", "None", type=str)
            if file_type == "Multi-Peak":
                # Create temporary plots and save them
                self.result_plot.show_spontaneous_peak_frequency(group_analysis)
                self._save_canvas(output_folder, "spontaneous_peak_frequency")

                window = self._amplitude_window_sec()
                self.result_plot.show_windowed_peak_amplitudes(group_analysis, window_sec=window)
                self._save_canvas(output_folder, f"peak_amplitudes_{window:g}s_windows")

            # Add more OutputManager plot saves as needed
        except Exception as e:
            QMessageBox.critical(
                self, "Export Failed",
                f"An error occurred while exporting:\n{e}"
            )
        else:
            msg = f"All metrics exported to:\n{output_folder}"
            if fit_failures:
                skipped = ", ".join(str(i + 1) for i, _reason in fit_failures)
                msg += (f"\n\nNote: the exponential fit could not be produced for "
                        f"{len(fit_failures)} timepoint(s) (file number {skipped}); "
                        f"those plots were skipped.")
            QMessageBox.information(self, "Plots Export Completed", msg)
        finally:
            progress.hide()

    def _ask_exponential_fit_timepoints(self, group_analysis):
        """Ask the user which timepoints should get an exponential-fit plot.

        Returns:
            list[int] | None: chosen file indices, or None if the user cancelled.
            An empty list means "export no fit plots", which is respected.
        """
        experiments = group_analysis.get_experiments()
        if not experiments:
            return []
        exp = experiments[0]
        times = group_analysis.get_timepoints_minutes()
        labels = []
        for i in range(exp.get_file_count()):
            fname = os.path.basename(exp.get_spheroid_file(i).get_filepath())
            labels.append(f"{times[i]:g} min  -  {fname}")

        dlg = TimepointMultiSelectionDialog(labels, self)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return dlg.get_selected_indices()
    
    def initializePage(self):
        """
        Called when the page is first displayed.

        Enables or disables analysis buttons depending on whether experiments are loaded.
        """
        if self.wizard().group_analysis.get_experiments():
            self.enable_analysis_buttons()
            # Check if spontaneous analysis should be shown
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            file_type = settings.value("file_type", "None", type=str)
            if file_type == "Multi-Peak":
                self.lbl_multi_peak.show()
                for btn in self.spontaneous_buttons:
                    btn.show()
                    btn.setEnabled(True)
            else:
                self.lbl_multi_peak.hide()
                for btn in self.spontaneous_buttons:
                    btn.hide()
        else:
            self.clear_all()

    def clear_all(self):
        """
        Clears the result plot and disables analysis buttons.
        """
        # Clear the plot
        self.result_plot.fig.clear()
        self.result_plot.draw()
        # Optionally disable analysis/export buttons
        for btn in getattr(self, 'analysis_buttons', []):
            btn.setEnabled(False)
        # Hide and disable multi-peak buttons and label
        if hasattr(self, 'lbl_multi_peak'):
            self.lbl_multi_peak.hide()
        for btn in getattr(self, 'spontaneous_buttons', []):
            btn.hide()
            btn.setEnabled(False)

    def enable_analysis_buttons(self):
        """
        Enables all buttons tied to group-level analysis or export functions.
        """
        for btn in getattr(self, 'analysis_buttons', []):
            btn.setEnabled(True)

    def handle_decay_fit(self):
        """
        Prompts the user to select a timepoint and displays the exponential decay fit
        for that timepoint using the group analysis results.
        """
        group_analysis = self.wizard().group_analysis
        experiments = group_analysis.get_experiments()
        if not experiments:
            return
        # Use the first experiment to get the list of files (timepoints)
        exp = experiments[0]
        file_names = [os.path.basename(exp.get_spheroid_file(i).get_filepath()) for i in range(exp.get_file_count())]
        dlg = TimepointSelectionDialog(file_names, self)
        if dlg.exec_() == QDialog.Accepted:
            idx = dlg.get_selected_index()
            self.result_plot.show_decay_exponential_fitting(group_analysis, replicate_time_point=idx)




