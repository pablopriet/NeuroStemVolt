from PyQt5.QtWidgets import (
    QApplication, QComboBox, QWizardPage, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QProgressDialog
)
from PyQt5.QtCore import QSettings, Qt

import os
from core.output_manager import OutputManager
from ui.utils.styles import apply_custom_styles
from ui.widgets.plot_canvas import PlotCanvas
from ui.wizard_2_colorplot.processing_dialog import ProcessingOptionsDialog

from core.output_manager import OutputManager
from core.processing import *

import numpy as np
import os

class ColorPlotPage(QWizardPage):
    """
    Wizard page for visualizing and processing color plot data from FSCV experiments.

    This page provides a user interface for:
    - Navigating between replicates and files.
    - Applying signal processing steps to data.
    - Visualizing processed data as color plots and I-T curves.
    - Saving or exporting results and visualizations.

    Attributes:
        selected_processors (list): List of user-selected processing steps.
        current_rep_index (int): Currently selected replicate index.
        current_file_index (int): Currently selected file index.
        main_plot (PlotCanvas): Canvas for rendering 2D color plots.
        it_plot (PlotCanvas): Canvas for rendering I-T profiles.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.setTitle("Color Plot")

        self.selected_processors = []
    
        # Left controls
        self.btn_revert = QPushButton("Reverse Changes")
        apply_custom_styles(self.btn_revert)
        self.btn_revert.clicked.connect(self.revert_processing)
        self.btn_eval = QPushButton("Evaluate")
        apply_custom_styles(self.btn_eval)
        self.btn_eval.clicked.connect(self.run_processing)

        self.cbo_rep = QComboBox(); 
        #apply_custom_styles(self.cbo_rep)
        self.cbo_rep.currentIndexChanged.connect(self.on_replicate_changed)
        
        #### Handle the signal from cbo_rep

        #self.txt_file = QLineEdit(); 
        #apply_custom_styles(self.txt_file)
        #self.txt_file.setReadOnly(True)

        self.cbo_file = QComboBox()
        self.cbo_file.currentIndexChanged.connect(self.on_file_changed)

        # Default indexes to visualize
        self.current_rep_index = 0
        self.current_file_index = 0

        self.btn_prev = QPushButton("Previous"); 
        apply_custom_styles(self.btn_prev)
        self.btn_next = QPushButton("Next")
        apply_custom_styles(self.btn_next)
        self.btn_prev.clicked.connect(self.on_prev_clicked)
        self.btn_next.clicked.connect(self.on_next_clicked)

        #### Handle the signal from prev and next btn

        self.btn_filter = QPushButton("Filter Options"); 
        apply_custom_styles(self.btn_filter)
        #btn_apply = QPushButton("Apply Filtering")
        self.btn_filter.clicked.connect(self.show_processing_options)
        self.btn_save = QPushButton("Save Current Plots"); 
        apply_custom_styles(self.btn_save)
        self.btn_save.clicked.connect(self.save_IT_ColorPlot_Plots)
        self.btn_export = QPushButton("Export Current IT")
        apply_custom_styles(self.btn_export)
        self.btn_export.clicked.connect(self.save_processed_data_IT)
        self.btn_export_all = QPushButton("Export All ITs")
        apply_custom_styles(self.btn_export_all)
        self.btn_export_all.clicked.connect(self.save_all_ITs)

        left = QVBoxLayout()
        left.addWidget(self.btn_revert)
        left.addWidget(self.cbo_rep)
        left.addWidget(self.cbo_file)

        nav = QHBoxLayout(); nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next)
    
        left.addLayout(nav)
        left.addWidget(self.btn_filter)
        left.addWidget(self.btn_eval)
        #left.addWidget(btn_apply)
        left.addWidget(self.btn_save)
        left.addWidget(self.btn_export)
        left.addWidget(self.btn_export_all)
        left.addStretch(1)

        # Right plots
        self.main_plot = PlotCanvas(self, width=5, height=4)

        self.it_plot = PlotCanvas(self, width=2.5, height=2)

        #cv_plot = PlotCanvas(self, width=2.5, height=2)
        #cv_plot.plot_line()

        bottom = QHBoxLayout()
        bottom.addWidget(self.it_plot)
        #bottom.addWidget(cv_plot)

        right = QVBoxLayout()
        right.addWidget(self.main_plot)
        right.addLayout(bottom)

        # Main layout for the page
        main_layout = QVBoxLayout()
        content_layout = QHBoxLayout()
        content_layout.addLayout(left)
        content_layout.addLayout(right)
        main_layout.addLayout(content_layout)

        # Footer
        footer = QLabel("© 2025 Hashemi Lab · NeuroStemVolt · v1.0.0")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("""
            color: gray;
            font-family: Helvetica, Arial;
            font-size: 10pt;
            margin-top: 12px;
        """)
        main_layout.addWidget(footer)

        self.setLayout(main_layout)
    
    def initializePage(self):
        """
        Initializes the page when it becomes visible.

        Loads the list of replicates and associated files into combo boxes.
        Enables/disables UI components depending on available data.
        """
        # Default index
        def_index = 0

        group_analysis = self.wizard().group_analysis
        display_names_list = self.wizard().display_names_list
        self.cbo_rep.clear()
        self.cbo_rep.addItems(display_names_list)
        self.cbo_rep.setCurrentIndex(def_index)
        self.cbo_rep.setEnabled(True)
        
        if not display_names_list:
            self.cbo_rep.setEnabled(False)
            self.cbo_file.clear()
            self.clear_all()
            return
        else:
            current_exp = group_analysis.get_single_experiments(def_index)
            file_names = [os.path.basename(current_exp.get_spheroid_file(i).get_filepath()) for i in range(current_exp.get_file_count())]
            self.cbo_file.clear()
            self.cbo_file.addItems(file_names)
            self.cbo_file.setCurrentIndex(0)
            self.cbo_file.setEnabled(True)
            self.update_file_display()

    def clear_all(self):
        """
        Clears all replicate and file selections and resets the plots.

        This method:
        - Resets internal indices to zero.
        - Clears both replicate and file combo boxes.
        - Clears the color plot and I-T canvas visuals.
        """
        # reset indices
        self.current_rep_index = 0
        self.current_file_index = 0

        # empty the combo box & filename display
        self.cbo_rep.clear()
        self.cbo_file.clear()

        # clear both canvases
        for canvas in (self.main_plot, self.it_plot):
            canvas.fig.clear()
            canvas.draw()

    def on_replicate_changed(self, index):
        """
        Handles replicate selection changes.

        Args:
            index (int): Index of the newly selected replicate.

        This updates the file combo box and visualizations accordingly.
        """
        self.current_rep_index = index
        self.current_file_index = 0
        self.update_file_display()

    def on_file_changed(self, index):
        """
        Handles file selection changes within a replicate.

        Args:
            index (int): Index of the newly selected file.
        """
        self.current_file_index = index
        self.update_file_display()

    def update_file_display(self):
        """
        Loads and displays the selected file's data.

        Updates both the color plot and I-T profile based on the
        current replicate and file selection. Handles out-of-bounds errors.
        """
        group_analysis = self.wizard().group_analysis
        try:
            exp = group_analysis.get_single_experiments(self.current_rep_index)
            sph_file = exp.get_spheroid_file(self.current_file_index)
            file_name = os.path.basename(sph_file.get_filepath())
            self.cbo_file.setCurrentText(file_name)

            processed_data = sph_file.get_processed_data()
            metadata = sph_file.get_metadata()
            peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position")

            self.main_plot.plot_color(processed_data=processed_data, peak_pos=peak_pos)
            self.it_plot.plot_IT(processed_data=processed_data, metadata=metadata, peak_position=peak_pos)
        except IndexError:
            self.cbo_file.setCurrentText("No file at this index")

    def on_next_clicked(self):
        """
        Advances to the next file in the current replicate, if available.
        """
        exp = self.wizard().group_analysis.get_single_experiments(self.current_rep_index)
        if self.current_file_index < exp.get_file_count() - 1:
            self.current_file_index += 1
            self.cbo_file.setCurrentIndex(self.current_file_index)

    def on_prev_clicked(self):
        """
        Moves to the previous file in the current replicate, if available.
        """
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.cbo_file.setCurrentIndex(self.current_file_index)

    def run_processing(self):
        """
        Runs the selected processing pipeline on the current experiment group.

        Automatically adds `FindAmplitude` as a mandatory step.
        Displays a progress dialog while processing.
        Updates visualizations after processing is complete.
        """
        group_analysis = self.wizard().group_analysis
        peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position", type=int)

        # Show loading dialog
        progress = QProgressDialog("Processing data, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()  # Ensure dialog appears

        # Always include FindAmplitude, but ensure it's not duplicated
        user_processors = self.selected_processors or []
        mandatory = FindAmplitude(peak_pos)

        processors = [p for p in user_processors if not isinstance(p, FindAmplitude)]
        processors.append(mandatory)

        group_analysis.set_processing_options_exp(processors)
        for exp in group_analysis.get_experiments():
            exp.run()

        self.update_file_display()
        progress.close()
        self.update_file_display()


    def revert_processing(self):
        group_analysis = self.wizard().group_analysis
        for exp in group_analysis.get_experiments():
            exp.revert_processing()
        self.update_file_display()

    def show_processing_options(self):
        """
        Opens the processing options dialog and updates selected processors.

        Retrieves the user’s choices and instantiates corresponding Processor objects.
        """
        dlg = ProcessingOptionsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            selected_names = dlg.get_selected_processors()
            peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position", type=int)
            self.selected_processors = [
                dlg.get_processor_instance(name, peak_pos)
                for name in selected_names
                if dlg.get_processor_instance(name, peak_pos) is not None
            ]

    def validatePage(self):
        """
        Validates this wizard page before proceeding.

        Applies `FindAmplitude` processing to ensure peak information is available
        for subsequent pages.

        Returns:
            bool: True if validation is successful, allowing transition to next page.
        """
        # Automatically add FindAmplitude processor and run it before proceeding
        group_analysis = self.wizard().group_analysis
        peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position", type=int)

        processor = FindAmplitude(peak_pos)

        for exp in group_analysis.get_experiments():
            exp.set_processing_steps([processor])  # only FindAmplitude
            exp.run()

        return True  # allow transition to next page

    def save_all_ITs(self):
        """
        Saves all I-T profiles from all experiments to the specified output folder.
        """
        group_analysis = self.wizard().group_analysis
        output_folder_path = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        OutputManager.save_all_ITs(group_analysis, output_folder_path)

    def save_IT_ColorPlot_Plots(self):
        """
        Saves the color plot and I-T profile visualizations for the current file.
        """
        exp = self.wizard().group_analysis.get_single_experiments(self.current_rep_index)
        sph_file = exp.get_spheroid_file(self.current_file_index)
        output_folder_path = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        sph_file.visualize_color_plot_data(title_suffix = "", save_path=output_folder_path)
        sph_file.visualize_IT_profile(QSettings("HashemiLab", "NeuroStemVolt").value("output_folder"))

    def save_processed_data_IT(self):
        """
        Saves the processed I-T data array (not a figure) for the current file.
        """
        exp = self.wizard().group_analysis.get_single_experiments(self.current_rep_index)
        sph_file = exp.get_spheroid_file(self.current_file_index)
        output_folder_path = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        OutputManager.save_IT_profile(sph_file,output_folder_path)
