from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QGridLayout, QDialog, QMessageBox, QProgressDialog, QApplication
)
from PyQt5.QtCore import QSettings, Qt

from core.output_manager import OutputManager
from core.processing import *

from ui.utils.styles import apply_custom_styles
from ui.widgets.plot_canvas import PlotCanvas
from ui.wizard_3_results.timepoint_dialog import (
    TimepointSelectionDialog, ExperimentSelectionDialog, 
    MultiExperimentTimepointDialog, MultiExperimentAmplitudeDialog,
    GroupedExperimentDialog, GroupedAmplitudeDialog, ViolinPlotDialog,
    get_experiment_display_name
)
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
        # prepare placeholders; actual analysis buttons are built in initializePage
        self.analysis = QGridLayout()
        self.analysis_buttons = []
        self.current_file_type = None  # track to detect changes

        # create plot canvas now (kept regardless of file_type)
        self.placeholder = QLabel("Select an analysis option to show plot")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.result_plot = PlotCanvas(self, width=5, height=4)
        self.result_plot.hide()  # start hidden

        # layout assembly
        main_layout = QVBoxLayout(self)

        # placeholder for analysis controls (will be populated in initializePage)
        main_layout.addLayout(self.analysis)

        # save/export - horizontal row
        save_export_layout = QHBoxLayout()
        btn_save     = QPushButton("Save Current Plot");          apply_custom_styles(btn_save)
        btn_save_all = QPushButton("Save All Plots");             apply_custom_styles(btn_save_all)
        btn_export   = QPushButton("Export metrics as csv");      apply_custom_styles(btn_export)
        save_export_layout.addWidget(btn_save)
        save_export_layout.addWidget(btn_save_all)
        save_export_layout.addWidget(btn_export)
        main_layout.addLayout(save_export_layout)

        # wire save/export buttons
        btn_save.clicked.connect(self.save_current_plot)
        btn_save_all.clicked.connect(self.save_all_plots)
        btn_export.clicked.connect(self.export_all_as_csv)

        # placeholder + future plot
        main_layout.addWidget(self.placeholder, stretch=1)
        main_layout.addWidget(self.result_plot, stretch=3)

        # footer
        footer = QLabel("© 2025 Hashemi Lab · NeuroStemVolt · v1.0.0")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: gray; font-size: 10pt;")
        main_layout.addWidget(footer)

        self.setLayout(main_layout)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                try:
                    w.setParent(None)
                    w.deleteLater()
                except Exception:
                    pass

    def _build_analysis_buttons(self, file_type):
        """
        Build and wire analysis buttons according to current file_type.
        """
        # clear existing
        self._clear_layout(self.analysis)
        self.analysis_buttons = []

        if file_type == "Spontaneous":
            btn_avg = QPushButton("Mean Amplitude Over Experiments"); apply_custom_styles(btn_avg)
            btn_freq = QPushButton("Frequency Over Time"); apply_custom_styles(btn_freq)
            btn_amp = QPushButton("Individual Amplitudes Over Time"); apply_custom_styles(btn_amp)
            self.analysis.addWidget(btn_avg, 0, 0)
            self.analysis.addWidget(btn_amp, 2, 0)
            self.analysis.addWidget(btn_freq, 0, 1)
            # connect
            btn_avg.clicked.connect(lambda _, f=lambda: self.result_plot.show_average_over_experiments(self.wizard().group_analysis): self._reveal_and_call(f))
            btn_amp.clicked.connect(lambda _, f=lambda: self.result_plot.show_amplitudes_over_time(self.wizard().group_analysis): self._reveal_and_call(f))
            btn_freq.clicked.connect(lambda: self._reveal_and_call(lambda: self.result_plot.show_frequency_over_time(self.wizard().group_analysis)))
            self.analysis_buttons = [btn_avg, btn_amp, btn_freq]
        else:
            # Stimulation file type - organized in 3x3 grid
            # Row 0: Amplitude metrics
            btn_avg = QPushButton("Mean Amplitude Over Experiments"); apply_custom_styles(btn_avg)
            btn_compare_amps = QPushButton("Compare Amplitudes (Multi-Exp)"); apply_custom_styles(btn_compare_amps)
            btn_compare_amps_grouped = QPushButton("Compare Amplitudes (Grouped)"); apply_custom_styles(btn_compare_amps_grouped)
            
            # Row 1: Tau/Decay metrics
            btn_param = QPushButton("Tau Over Time"); apply_custom_styles(btn_param)
            btn_fit = QPushButton("Decay Exponential Fitting"); apply_custom_styles(btn_fit)
            btn_compare_tau_grouped = QPushButton("Compare Tau (Grouped)"); apply_custom_styles(btn_compare_tau_grouped)
            
            # Row 2: IT and distribution plots
            btn_it_series = QPushButton("IT Time Series (Single Exp)"); apply_custom_styles(btn_it_series)
            btn_compare_its = QPushButton("Compare ITs at Timepoint"); apply_custom_styles(btn_compare_its)
            btn_violin_plot = QPushButton("Amplitude Distribution (Violin)"); apply_custom_styles(btn_violin_plot)
            
            # Layout: 3x3 grid
            self.analysis.addWidget(btn_avg, 0, 0)
            self.analysis.addWidget(btn_compare_amps, 0, 1)
            self.analysis.addWidget(btn_compare_amps_grouped, 0, 2)
            self.analysis.addWidget(btn_param, 1, 0)
            self.analysis.addWidget(btn_fit, 1, 1)
            self.analysis.addWidget(btn_compare_tau_grouped, 1, 2)
            self.analysis.addWidget(btn_it_series, 2, 0)
            self.analysis.addWidget(btn_compare_its, 2, 1)
            self.analysis.addWidget(btn_violin_plot, 2, 2)
            
            # connect
            btn_avg.clicked.connect(lambda _, f=lambda: self.result_plot.show_average_over_experiments(self.wizard().group_analysis): self._reveal_and_call(f))
            btn_fit.clicked.connect(lambda _, f=self.handle_decay_fit: self._reveal_and_call(f))
            btn_param.clicked.connect(lambda _, f=lambda: self.result_plot.show_tau_param_over_time(self.wizard().group_analysis): self._reveal_and_call(f))
            btn_it_series.clicked.connect(lambda: self._reveal_and_call(self.handle_it_time_series))
            btn_compare_its.clicked.connect(lambda: self._reveal_and_call(self.handle_compare_its_at_timepoint))
            btn_compare_amps.clicked.connect(lambda: self._reveal_and_call(self.handle_compare_amplitudes_multi_exp))
            btn_compare_tau_grouped.clicked.connect(lambda: self._reveal_and_call(self.handle_compare_tau_grouped))
            btn_compare_amps_grouped.clicked.connect(lambda: self._reveal_and_call(self.handle_compare_amplitudes_grouped))
            btn_violin_plot.clicked.connect(lambda: self._reveal_and_call(self.handle_violin_plot))
            
            self.analysis_buttons = [btn_avg, btn_compare_amps, btn_param, btn_fit, btn_it_series, 
                                    btn_compare_its, btn_compare_tau_grouped, btn_compare_amps_grouped, btn_violin_plot]

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
            # Export spontaneous peak metrics if applicable
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            file_type = settings.value("file_type", "None", type=str)
            print(file_type)
            print(type(file_type))
            
            # Generate comprehensive experiment log
            OutputManager.save_experiment_log(ga, output_folder, qsettings=settings)
            
            if file_type == "Spontaneous":
                OutputManager.save_spontaneous_peak_metrics(ga, output_folder)
            else:
                OutputManager.save_all_peak_amplitudes(ga, output_folder)
                OutputManager.save_all_reuptake_curves(ga, output_folder)
                OutputManager.save_all_exponential_fitting_params(ga, output_folder)
                OutputManager.save_all_exponential_fitting_params_global(ga, output_folder)
                OutputManager.save_all_AUC(ga, output_folder)
            OutputManager.save_all_ITs(ga, output_folder)
            OutputManager.save_mean_ITs(ga, output_folder)


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

        Prompts the user for a filename using QFileDialog.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current Plot As",
            "",
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if file_path:
            self.result_plot.fig.savefig(file_path, dpi=300, bbox_inches='tight')
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
        settings = (QSettings("HashemiLab", "NeuroStemVolt"))
        output_folder = settings.value("output_folder")
        file_type = settings.value("file_type", "None", type=str)
        group_analysis = self.wizard().group_analysis
        if not output_folder or not os.path.isdir(output_folder):
            QMessageBox.warning(self, "No Output Folder", "Please set a valid output folder in Experiment Settings.")
            return

        # create & show the indeterminate progress dialog
        progress = QProgressDialog("Exporting plots...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)    # hide the cancel button
        progress.setMinimumDuration(0)    # show immediately
        progress.setAutoClose(False)
        progress.show()
        QApplication.processEvents()      # force a repaint

        try:
            # Save all group-level plots using OutputManager
            OutputManager.save_mean_ITs_plot(group_analysis, output_folder)
            OutputManager.save_plot_all_amplitudes_over_time(group_analysis, output_folder)
            OutputManager.save_plot_mean_amplitudes_over_time(group_analysis, output_folder)

            if not file_type == "Spontaneous":
                OutputManager.save_plot_tau_over_time(group_analysis, output_folder)
                OutputManager.save_plot_exponential_fit_aligned(group_analysis, output_folder)
                
                # Diagnostic plots to showcase processing features
                OutputManager.save_diagnostic_AUC_plot(group_analysis, output_folder)
                OutputManager.save_diagnostic_reuptake_fits_plot(group_analysis, output_folder)
            else:
                OutputManager.save_plot_frequency_over_time(group_analysis, output_folder)
                
                # Processing diagnostic plots (applicable to all file types)
                OutputManager.save_diagnostic_butterworth_filter_plot(group_analysis, output_folder)
                OutputManager.save_diagnostic_processing_pipeline_plot(group_analysis, output_folder)
                OutputManager.save_diagnostic_filter_comparison_plot(group_analysis, output_folder)
                OutputManager.save_diagnostic_peak_detection_plot(group_analysis, output_folder)
                OutputManager.save_diagnostic_color_plot_comparison(group_analysis, output_folder)
        except Exception as e:
            QMessageBox.critical(
                self, "Export Failed",
                f"An error occurred while exporting:\n{e}"
            )
        else:
            QMessageBox.information(
                self, "Plots Export Completed",
                f"All metrics exported to:\n{output_folder}"
            )
        finally:
            progress.hide()

    def initializePage(self):
        """
        Called when the page is first displayed.

        Enables or disables analysis buttons depending on whether experiments are loaded.
        """
        # rebuild analysis UI if file_type changed fore xample whether spontaenous vs estimulated 
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        file_type = settings.value("file_type", "None", type=str)
        if file_type != self.current_file_type:
            self.current_file_type = file_type
            self._build_analysis_buttons(file_type)

        if self.wizard().group_analysis.get_experiments():
            self.enable_analysis_buttons()
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

    def handle_amplitude_single_file(self):
        """
        Prompts the user to select a timepoint (file) and displays the amplitude analysis
        for that single file using the group analysis results.
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
            # Try a dedicated plotting method if available; otherwise, fall back or warn
            try:
                plotter = getattr(self.result_plot, 'show_amplitude_for_single_file')
                plotter(group_analysis, replicate_time_point=idx)
            except AttributeError:
                # Fallback: if no dedicated method exists, try reusing amplitudes-over-time and focus a single trace
                try:
                    self.result_plot.show_amplitudes_for_timepoint(group_analysis, replicate_time_point=idx)
                except AttributeError:
                    QMessageBox.warning(self, "Not Implemented", "Single-file amplitude plot is not implemented in PlotCanvas.")

    def handle_it_time_series(self):
        """
        Display consecutive ITs from a single experiment with a color gradient.
        The color intensity increases with time within the experiment.
        If calibration is enabled, displays concentration vs time instead of current vs time.
        """
        group_analysis = self.wizard().group_analysis
        experiments = group_analysis.get_experiments()
        if not experiments:
            QMessageBox.warning(self, "No Data", "No experiments loaded.")
            return
        
        # Create experiment names for selection dialog using folder names
        experiment_names = [get_experiment_display_name(exp, i) for i, exp in enumerate(experiments)]
        
        # Show selection dialog
        dlg = ExperimentSelectionDialog(experiment_names, self, title="Select Experiment for IT Time Series")
        if dlg.exec_() == QDialog.Accepted:
            exp_idx = dlg.get_selected_index()
            self.result_plot.plot_it_time_series_single_experiment(group_analysis, exp_idx)

    def handle_amplitudes_vs_timepoint(self):
        """
        LEGACY: Display peak amplitudes across timepoints/files for a single stimulation experiment.
        Shows one amplitude value per file, plotted against file index or time.
        Only works for stimulation files (not spontaneous).
        
        NOTE: This function has been deprecated in favor of the grouped comparison plots.
        Kept for backward compatibility but UI button has been removed.
        """
        group_analysis = self.wizard().group_analysis
        experiments = group_analysis.get_experiments()
        if not experiments:
            QMessageBox.warning(self, "No Data", "No experiments loaded.")
            return
        
        # Check file type - only for stimulation files
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        file_type = settings.value("file_type", "None", type=str)
        if file_type == "Spontaneous":
            QMessageBox.warning(self, "Invalid File Type", 
                              "This plot is only available for stimulation files, not spontaneous recordings.")
            return
        
        # Create experiment names for selection dialog using folder names
        experiment_names = [get_experiment_display_name(exp, i) for i, exp in enumerate(experiments)]
        
        # Show selection dialog
        dlg = ExperimentSelectionDialog(experiment_names, self, title="Select Experiment for Amplitude Timeline")
        if dlg.exec_() == QDialog.Accepted:
            exp_idx = dlg.get_selected_index()
            self.result_plot.plot_amplitudes_vs_timepoint_single_experiment_legacy(group_analysis, exp_idx)

    def handle_compare_its_at_timepoint(self):
        """
        Compare IT traces from multiple experiments at a specific timepoint.
        User selects:
        1. A specific timepoint (e.g., -30 min, 0 min, 10 min)
        2. Multiple experiments to compare
        3. Custom colors for each experiment
        """
        group_analysis = self.wizard().group_analysis
        experiments = group_analysis.get_experiments()
        if not experiments:
            QMessageBox.warning(self, "No Data", "No experiments loaded.")
            return
        
        # Get time parameters from first experiment
        first_exp = experiments[0]
        time_between_files = first_exp.get_time_between_files()
        files_before_treatment = first_exp.get_number_of_files_before_treatment()
        
        # Show the multi-experiment timepoint selection dialog
        dlg = MultiExperimentTimepointDialog(
            experiments, time_between_files, files_before_treatment, self
        )
        
        if dlg.exec_() == QDialog.Accepted:
            timepoint_idx = dlg.get_selected_timepoint_index()
            timepoint_label = dlg.get_selected_timepoint_label()
            selected_experiments = dlg.get_selected_experiments()  # List of (exp_idx, color)
            
            if not selected_experiments:
                QMessageBox.warning(self, "No Selection", "Please select at least one experiment.")
                return
            
            self.result_plot.plot_compare_its_at_timepoint(
                group_analysis, timepoint_idx, timepoint_label, selected_experiments
            )

    def handle_compare_amplitudes_multi_exp(self):
        """
        Compare amplitude timelines from multiple experiments.
        User selects:
        1. Multiple experiments to compare
        2. Custom colors for each experiment
        """
        group_analysis = self.wizard().group_analysis
        experiments = group_analysis.get_experiments()
        if not experiments:
            QMessageBox.warning(self, "No Data", "No experiments loaded.")
            return
        
        # Check file type - only for stimulation files
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        file_type = settings.value("file_type", "None", type=str)
        if file_type == "Spontaneous":
            QMessageBox.warning(self, "Invalid File Type", 
                              "This plot is only available for stimulation files, not spontaneous recordings.")
            return
        
        # Show the multi-experiment amplitude selection dialog
        dlg = MultiExperimentAmplitudeDialog(experiments, self)
        
        if dlg.exec_() == QDialog.Accepted:
            selected_experiments = dlg.get_selected_experiments()  # List of (exp_idx, color)
            
            if not selected_experiments:
                QMessageBox.warning(self, "No Selection", "Please select at least one experiment.")
                return
            
            self.result_plot.plot_compare_amplitudes_multi_exp(group_analysis, selected_experiments)

    def handle_compare_tau_grouped(self):
        """
        Compare tau values from grouped experiments at a specific timepoint.
        
        User selects:
        1. A specific timepoint
        2. Multiple groups with experiments in each
        3. Custom colors for each group
        
        Experiments in the same group are averaged together.
        """
        group_analysis = self.wizard().group_analysis
        experiments = group_analysis.get_experiments()
        if not experiments:
            QMessageBox.warning(self, "No Data", "No experiments loaded.")
            return
        
        # Get time parameters from first experiment
        first_exp = experiments[0]
        time_between_files = first_exp.get_time_between_files()
        files_before_treatment = first_exp.get_number_of_files_before_treatment()
        
        # Show the grouped experiment dialog
        dlg = GroupedExperimentDialog(
            experiments, time_between_files, files_before_treatment, 
            self, title="Compare Tau Across Groups", metric_name="Tau"
        )
        
        if dlg.exec_() == QDialog.Accepted:
            timepoint_idx = dlg.get_selected_timepoint_index()
            timepoint_label = dlg.get_selected_timepoint_label()
            groups = dlg.get_groups()
            
            if not groups:
                QMessageBox.warning(self, "No Groups", "Please create at least one group with experiments.")
                return
            
            self.result_plot.plot_compare_tau_grouped(
                group_analysis, timepoint_idx, timepoint_label, groups
            )

    def handle_compare_amplitudes_grouped(self):
        """
        Compare amplitudes from grouped experiments at a specific timepoint.
        
        User selects:
        1. A specific timepoint
        2. Multiple groups with experiments in each
        3. Custom colors for each group
        
        Experiments in the same group are averaged together.
        """
        group_analysis = self.wizard().group_analysis
        experiments = group_analysis.get_experiments()
        if not experiments:
            QMessageBox.warning(self, "No Data", "No experiments loaded.")
            return
        
        # Get time parameters from first experiment
        first_exp = experiments[0]
        time_between_files = first_exp.get_time_between_files()
        files_before_treatment = first_exp.get_number_of_files_before_treatment()
        
        # Show the grouped amplitude dialog (single timepoint)
        dlg = GroupedAmplitudeDialog(
            experiments, time_between_files, files_before_treatment, 
            self, title="Compare Amplitudes Across Groups"
        )
        
        if dlg.exec_() == QDialog.Accepted:
            timepoint_idx = dlg.get_selected_timepoint_index()
            timepoint_label = dlg.get_selected_timepoint_label()
            groups = dlg.get_groups()
            
            if not groups:
                QMessageBox.warning(self, "No Groups", "Please create at least one group with experiments.")
                return
            
            self.result_plot.plot_compare_amplitudes_grouped(
                group_analysis, timepoint_idx, timepoint_label, groups
            )

    def handle_violin_plot(self):
        """
        Create a violin/swarm plot comparing amplitude distributions between groups.
        
        User selects:
        1. Exactly two timepoints (for difference calculation or comparison)
        2. Multiple groups with experiments in each
        3. Custom colors for each group
        4. Custom plot title and y-axis label
        
        Creates a violin plot with overlaid swarm points showing distribution.
        """
        group_analysis = self.wizard().group_analysis
        experiments = group_analysis.get_experiments()
        if not experiments:
            QMessageBox.warning(self, "No Data", "No experiments loaded.")
            return
        
        # Get time parameters from first experiment
        first_exp = experiments[0]
        time_between_files = first_exp.get_time_between_files()
        files_before_treatment = first_exp.get_number_of_files_before_treatment()
        
        # Show the violin plot dialog
        dlg = ViolinPlotDialog(
            experiments, time_between_files, files_before_treatment, 
            self, title="Amplitude Distribution Comparison"
        )
        
        if dlg.exec_() == QDialog.Accepted:
            timepoints = dlg.get_selected_timepoints()
            groups = dlg.get_groups()
            plot_title = dlg.get_plot_title()
            ylabel = dlg.get_ylabel()
            
            if not groups:
                QMessageBox.warning(self, "No Groups", "Please create at least one group with experiments.")
                return
            
            self.result_plot.plot_amplitude_violin(
                group_analysis, timepoints, groups, plot_title, ylabel
            )
