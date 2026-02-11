from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QPushButton, QVBoxLayout, QFileDialog, QGridLayout, QDialog, QMessageBox, QProgressDialog, QApplication
)
from PyQt5.QtCore import QSettings, Qt

from core.output_manager import OutputManager
from core.processing import *

from ui.utils.styles import apply_custom_styles
from ui.widgets.plot_canvas import PlotCanvas
from ui.wizard_3_results.timepoint_dialog import TimepointSelectionDialog
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

        # analysis buttons
        btn_avg    = QPushButton("Mean Amplitude Over Experiments");      apply_custom_styles(btn_avg)
        btn_fit    = QPushButton("Decay Exponential Fitting");     apply_custom_styles(btn_fit)
        btn_param  = QPushButton("Tau Over Time");                 apply_custom_styles(btn_param)
        btn_amp    = QPushButton("Individual Amplitudes Over Time");          apply_custom_styles(btn_amp)

        # Spontaneous analysis buttons
        self.btn_spont_freq = QPushButton("Spontaneous Peak Frequency");     apply_custom_styles(self.btn_spont_freq)
        self.btn_spont_amp = QPushButton("Spontaneous Peak Amplitudes");     apply_custom_styles(self.btn_spont_amp)
        self.spontaneous_buttons = [self.btn_spont_freq, self.btn_spont_amp]

        # Initially hide spontaneous buttons
        for btn in self.spontaneous_buttons:
            btn.hide()

        # grid of analysis buttons
        analysis = QGridLayout()
        analysis.addWidget(btn_avg,    0, 0)
        analysis.addWidget(btn_fit,    1, 1)
        analysis.addWidget(btn_param,  1, 0)
        analysis.addWidget(btn_amp,    0, 1)
        analysis.addWidget(self.btn_spont_freq, 2, 0)
        analysis.addWidget(self.btn_spont_amp, 2, 1)

        # save/export buttons
        btn_save     = QPushButton("Save Current Plot");          apply_custom_styles(btn_save)
        btn_save_all = QPushButton("Save All Plots");             apply_custom_styles(btn_save_all)
        btn_export   = QPushButton("Export metrics as csv");      apply_custom_styles(btn_export)

        self.analysis_buttons = [btn_avg, btn_fit, btn_param, btn_amp, btn_save, btn_save_all, btn_export]

        # placeholder & plotCanvas
        self.placeholder = QLabel("Select an analysis option to show plot")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.result_plot = PlotCanvas(self, width=5, height=4)
        self.result_plot.hide()  # start hidden

        # connect buttons to show‐and‐plot
        for btn, fn in (
            (btn_avg,   lambda: self.result_plot.show_average_over_experiments(self.wizard().group_analysis)),
            (btn_fit,   self.handle_decay_fit),
            (btn_param, lambda: self.result_plot.show_tau_param_over_time(self.wizard().group_analysis)),
            (btn_amp,   lambda: self.result_plot.show_amplitudes_over_time(self.wizard().group_analysis)),
            (self.btn_spont_freq, lambda: self.result_plot.show_spontaneous_peak_frequency(self.wizard().group_analysis)),
            (self.btn_spont_amp, lambda: self.result_plot.show_spontaneous_peak_amplitudes(self.wizard().group_analysis)),
        ):
            btn.clicked.connect(lambda _, f=fn: self._reveal_and_call(f))

        btn_save.clicked.connect(self.save_current_plot)
        btn_save_all.clicked.connect(self.save_all_plots)
        btn_export.clicked.connect(self.export_all_as_csv)

        # layout assembly
        main_layout = QVBoxLayout(self)

        # analysis buttons at top
        main_layout.addLayout(analysis)

        # save/export
        main_layout.addWidget(btn_save)
        main_layout.addWidget(btn_save_all)
        main_layout.addWidget(btn_export)

        # placeholder + future plot
        main_layout.addWidget(self.placeholder, stretch=1)
        main_layout.addWidget(self.result_plot, stretch=3)

        # footer
        footer = QLabel("© 2025 Hashemi Lab · NeuroStemVolt · v1.0.0")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: gray; font-size: 10pt;")
        main_layout.addWidget(footer)

        self.setLayout(main_layout)

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
            OutputManager.save_all_exponential_fitting_params(ga, output_folder)
            OutputManager.save_all_AUC(ga, output_folder)
            OutputManager.save_mean_ITs(ga, output_folder)
            #OutputManager.save_mean_processed_data_matrices(ga,output_folder)
            
            # Export spontaneous peak metrics if applicable
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            file_type = settings.value("file_type", "None", type=str)
            if file_type == "Spontaneous":
                OutputManager.save_spontaneous_peak_metrics(ga, output_folder)
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
        output_folder = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
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
            OutputManager.save_plot_tau_over_time(group_analysis, output_folder)
            OutputManager.save_plot_exponential_fit_aligned(group_analysis, output_folder)
            OutputManager.save_plot_all_amplitudes_over_time(group_analysis, output_folder)
            OutputManager.save_plot_mean_amplitudes_over_time(group_analysis, output_folder)

            # Save spontaneous analysis plots if applicable
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            file_type = settings.value("file_type", "None", type=str)
            if file_type == "Spontaneous":
                # Create temporary plots and save them
                self.result_plot.show_spontaneous_peak_amplitudes(group_analysis)
                self.result_plot.fig.savefig(os.path.join(output_folder, "spontaneous_peak_amplitudes.png"), 
                                           dpi=300, bbox_inches='tight')

                self.result_plot.show_spontaneous_peak_frequency(group_analysis)
                self.result_plot.fig.savefig(os.path.join(output_folder, "spontaneous_peak_frequency.png"),
                                           dpi=300, bbox_inches='tight')

            # Add more OutputManager plot saves as needed
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
        if self.wizard().group_analysis.get_experiments():
            self.enable_analysis_buttons()
            # Check if spontaneous analysis should be shown
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            file_type = settings.value("file_type", "None", type=str)
            if file_type == "Spontaneous":
                for btn in self.spontaneous_buttons:
                    btn.show()
                    btn.setEnabled(True)
            else:
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
        # Hide and disable spontaneous buttons
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




