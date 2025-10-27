from PyQt5.QtWidgets import (
    QApplication, QComboBox, QWizardPage, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QProgressDialog, QSlider, QToolTip
)
from PyQt5.QtCore import QSettings, Qt, QEvent

import os
from core.output_manager import OutputManager
from ui.utils.styles import apply_custom_styles
from ui.widgets.plot_canvas import PlotCanvas
from ui.wizard_2_colorplot.processing_dialog import ProcessingOptionsDialog

from core.output_manager import OutputManager
from core.processing import *
# Import both amplitude processors at module level
from core.processing.find_amplitude import FindAmplitude
from core.processing.spontaneous_peak_detector import FindAmplitudeMultiple

import numpy as np
import os
import re

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
        self.file_index_mapping = []  # Add this to track sorted file indices
        self.temp_peak = None

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
        self.btn_adj_peak = QPushButton("Apply Peak Adjustment")
        apply_custom_styles(self.btn_adj_peak)
        self.btn_adj_peak.clicked.connect(self.adjust_peak_position)
        for b in (self.btn_prev, self.btn_next, self.btn_eval, self.btn_filter,
          self.btn_save, self.btn_export, self.btn_export_all, self.btn_adj_peak):
            b.setAutoDefault(False)
            b.setDefault(False)

        self.peak_slider = QSlider(Qt.Orientation.Horizontal)
        self.peak_slider.setMinimum(0)
        self.peak_slider.setMaximum(600)
        self.peak_slider.setValue(50)
        self.peak_slider.setTickInterval(1)
        self.peak_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.peak_slider.valueChanged.connect(self.on_peak_det_slider_value_changed)
        self.peak_slider.setMaximumWidth(400)
        self.peak_slider.setFixedHeight(30)

        self._set_peak_controls_enabled(False)

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
        peak_label = QLabel("Peak Adjustment")
        peak_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        left.addWidget(peak_label)
        left.addWidget(self.peak_slider)
        left.addWidget(self.btn_adj_peak)


        # Right plots
        self.main_plot = PlotCanvas(self, width=5, height=4)

        self.it_plot = PlotCanvas(self, width=4, height=3)

        file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)
        if file_type == "Spontaneous":
            # Add CV plot canvas
            self.cv_plot = PlotCanvas(self, width=2.5, height=2)

        bottom = QHBoxLayout()
        bottom.addWidget(self.it_plot)
        if file_type == "Spontaneous":
            bottom.addWidget(self.cv_plot)  # Add the CV plot here

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

    def natural_sort_key(self, filename):
        """Natural sorting key that handles numbers properly."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

    def initializePage(self):
        """
        Initializes the page when it becomes visible.

        Loads the list of replicates and associated files into combo boxes.
        Enables/disables UI components depending on available data.
        """
        super().initializePage()
        # Default index
        def_index = 0

        group_analysis = self.wizard().group_analysis
        display_names_list = self.wizard().display_names_list
        self.cbo_rep.clear()
        self.cbo_rep.addItems(display_names_list)
        self.cbo_rep.setCurrentIndex(def_index)
        self.cbo_rep.setEnabled(True)

        # Grab the Next button from the wizard
        next_btn = self.wizard().button(self.wizard().NextButton)
        next_btn.setToolTip("Press 'Evaluate' to find peaks before continuing.")
        next_btn.setAttribute(Qt.WA_AlwaysShowToolTips, True)  # show even if disabled
        next_btn.installEventFilter(self)

        if not display_names_list:
            self.cbo_rep.setEnabled(False)
            self.cbo_file.clear()
            self.clear_all()
            return
        else:
            self._update_file_list_for_replicate(def_index)
            self.update_file_display()

    def eventFilter(self, obj, event):
        if obj is self.wizard().button(self.wizard().NextButton):
            if event.type() == QEvent.ToolTip:
                if not self.isComplete():  # Only show when Next is disabled
                    QToolTip.showText(event.globalPos(),
                                    "You need to press 'Evaluate' first to detect peaks.")
                    return True  # block default
        return super().eventFilter(obj, event)

    def _update_file_list_for_replicate(self, rep_index):
        """Update the file dropdown and mapping for a specific replicate."""
        group_analysis = self.wizard().group_analysis
        current_exp = group_analysis.get_single_experiments(rep_index)

        # Create list of (filename, original_index) pairs
        file_info = [(os.path.basename(current_exp.get_spheroid_file(i).get_filepath()), i)
                    for i in range(current_exp.get_file_count())]

        # Sort by filename using natural sorting
        file_info.sort(key=lambda x: self.natural_sort_key(x[0]))

        # Extract sorted filenames and create index mapping
        file_names = [info[0] for info in file_info]
        self.file_index_mapping = [info[1] for info in file_info]

        # Temporarily disconnect signal to avoid recursion
        self.cbo_file.currentIndexChanged.disconnect()

        self.cbo_file.clear()
        self.cbo_file.addItems(file_names)
        self.cbo_file.setCurrentIndex(0)
        self.cbo_file.setEnabled(True)

        # Reconnect signal
        self.cbo_file.currentIndexChanged.connect(self.on_file_changed)

    def clear_all(self):
        """
        Clears all replicate and file selections and resets the plots.

        This method:
        - Resets internal indices to zero.
        - Clears both replicate and file combo boxes.
        - Clears the color plot and I-T canvas visuals.
        """
        self.current_rep_index = 0
        self.current_file_index = 0
        self.cbo_rep.clear()
        self.cbo_file.clear()

        canvases = [self.main_plot, self.it_plot]
        if hasattr(self, "cv_plot"):
            canvases.append(self.cv_plot)

        for canvas in canvases:
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
        self._update_file_list_for_replicate(index)
        self.update_file_display()
        self._set_peak_controls_enabled(self.isComplete())

    def on_file_changed(self, index):
        """
        Handles file selection changes within a replicate.

        Args:
            index (int): Index of the newly selected file.
        """
        self.current_file_index = index
        self.update_file_display()
        self._set_peak_controls_enabled(self.isComplete())

    def update_file_display(self):
        """
        Loads and displays the selected file's data.

        Updates both the color plot and I-T profile based on the
        current replicate and file selection. Handles out-of-bounds errors.
        """
        group_analysis = self.wizard().group_analysis
        try:
            exp = group_analysis.get_single_experiments(self.current_rep_index)

            # Use the mapping to get the actual file index
            if (self.file_index_mapping and
                self.current_file_index < len(self.file_index_mapping)):
                actual_file_index = self.file_index_mapping[self.current_file_index]
            else:
                # Fallback if mapping is not available
                actual_file_index = self.current_file_index

            sph_file = exp.get_spheroid_file(actual_file_index)

            # DON'T call setCurrentText here - it causes recursion!
            # The dropdown is already showing the correct text

            processed_data = sph_file.get_processed_data()
            metadata = sph_file.get_metadata()
            peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position")

            self.main_plot.plot_color(processed_data=processed_data, peak_pos=peak_pos)

            self.it_plot.plot_IT(processed_data=processed_data, metadata=metadata, peak_position=peak_pos,
                                 temp_peak_detection=self.temp_peak)

            file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)
            if file_type == "Spontaneous":
                # Plot CV(s) correlated with detected peaks
                self.cv_plot.plot_cv(processed_data=processed_data, metadata=metadata,
                                   title_suffix=f"File {self.current_file_index + 1}")

        except IndexError:
            # Handle error case
            print(f"Error: Cannot access file at index {self.current_file_index}")

    def on_next_clicked(self):
        # Use the mapping length instead of original file count
        max_files = len(self.file_index_mapping) if self.file_index_mapping else 0

        if self.current_file_index < max_files - 1:
            self.current_file_index += 1
            self.cbo_file.setCurrentIndex(self.current_file_index)

    def on_prev_clicked(self):
        """
        Moves to the previous file in the current replicate, if available.
        """
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.cbo_file.setCurrentIndex(self.current_file_index)

    def on_peak_det_slider_value_changed(self, changed_value):
        if not self.peak_slider.isEnabled() or not self.isComplete():
            return
        self.temp_peak = changed_value
        self.update_file_display()

    def adjust_peak_position(self):
        if self.temp_peak is not None:
            try:
                group_analysis = self.wizard().group_analysis
                if group_analysis and group_analysis.get_experiments():
                    current_exp = group_analysis.get_single_experiments(self.current_rep_index)
                    if current_exp:
                        print(f"DEBUG: Adjusting to voltage position {self.temp_peak}")

                        # Use the mapping to get the actual file index
                        if (self.file_index_mapping and
                                self.current_file_index < len(self.file_index_mapping)):
                            actual_file_index = self.file_index_mapping[self.current_file_index]
                        else:
                            # Fallback if mapping is not available
                            actual_file_index = self.current_file_index
                        # Get the current file
                        current_file = current_exp.get_spheroid_file(actual_file_index)

                        # Get the processed data to calculate the new peak value
                        processed_data = current_file.get_processed_data()
                        if processed_data is not None and self.temp_peak < processed_data.shape[1]:
                            # Get the IT profile at the new voltage position

                            new_peak_value = 0
                            metadata = current_file.get_metadata()
                            new_peak_time_index = self.temp_peak
                            new_peak_value = processed_data[new_peak_time_index, metadata["peak_position"]]


                            print(
                                f"DEBUG: At voltage {self.temp_peak}: peak_value={new_peak_value:.3f}, time_index={new_peak_time_index}")
                        else:
                            new_peak_value = 0.0
                            new_peak_time_index = 0

                        # Update metadata
                        update_dict = {
                            'peak_amplitude_positions': new_peak_time_index,
                            'peak_amplitude_values': new_peak_value
                        }
                        current_file.update_metadata(update_dict)
                        print(f"DEBUG: Updated metadata - positions: {new_peak_time_index}, values: {new_peak_value}")

                        # **DIRECTLY UPDATE THE PLOTS WITHOUT CALLING update_file_display()**
                        # This prevents any processing pipeline from overwriting our manual metadata
                        try:
                            peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position")

                            # Get the CURRENT metadata (which should have our manual values)
                            current_metadata = current_file.get_metadata()
                            print(
                                f"DEBUG: Using metadata for display: positions={current_metadata.get('peak_amplitude_positions')}, values={current_metadata.get('peak_amplitude_values')}")

                            # Update plots directly
                            self.main_plot.plot_color(processed_data=processed_data, peak_pos=peak_pos)
                            self.it_plot.plot_IT(processed_data=processed_data, metadata=current_metadata,
                                                 peak_position=peak_pos)

                            file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)
                            if file_type == "Spontaneous":
                                self.cv_plot.plot_cv(processed_data=processed_data, metadata=current_metadata,
                                                     title_suffix=f"File {self.current_file_index + 1}")

                            print("DEBUG: Plots updated directly with manual metadata")

                        except Exception as plot_error:
                            print(f"Error updating plots directly: {plot_error}")
                            # Fallback to normal update if direct update fails
                            self.update_file_display()

                        print(
                            f"Applied peak amplitude update at voltage position: {self.temp_peak}, time index: {new_peak_time_index}, value: {new_peak_value:.3f}")

                        # Reset temp value
                        self.temp_peak = None
            except Exception as e:
                print(f"Error applying peak detection change: {e}")
                import traceback
                traceback.print_exc()
        self.completeChanged.emit()

    def run_processing(self):
        """
        Runs the selected processing pipeline on the current experiment group.

        Automatically adds `FindAmplitude` as a mandatory step.
        Displays a progress dialog while processing.
        Updates visualizations after processing is complete.
        """
        self.revert_processing() #<- Default revert to raw data before processing to avoid cumulative effects
        group_analysis = self.wizard().group_analysis
        peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position", type=int)

        # Check file type to determine which amplitude finder to use
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        file_type = settings.value("file_type", "None", type=str)

        # Show loading dialog
        progress = QProgressDialog("Processing data, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        # Choose the appropriate amplitude finder based on file type
        if file_type == "Spontaneous":
            mandatory = FindAmplitudeMultiple(peak_pos)
        else:
            print("Using default amplitude finder______________")
            mandatory = FindAmplitude(peak_pos)

        # Keep all user processors EXCEPT any existing amplitude finders
        user_processors = self.selected_processors or []
        processors = [p for p in user_processors if not isinstance(p, (FindAmplitude))]

        # Add the mandatory amplitude finder
        processors.append(mandatory)
        print(processors)

        group_analysis.set_processing_options_exp(processors)
        for exp in group_analysis.get_experiments():
            exp.run()

        self.update_file_display()
        self.completeChanged.emit()
        self._set_peak_controls_enabled(self.isComplete())
        progress.close()


    def revert_processing(self):
        group_analysis = self.wizard().group_analysis
        for exp in group_analysis.get_experiments():
            exp.revert_processing()
        # disarm the slider & clear temp point
        self._set_peak_controls_enabled(False)
        self.update_file_display()
        self.completeChanged.emit()

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

    def _missing_peaks(self):
        """Return a list of (rep_index, file_index) that do not have peak metadata."""
        missing = []
        group_analysis = self.wizard().group_analysis
        for r_idx, exp in enumerate(group_analysis.get_experiments()):
            file_count = exp.get_file_count()
            for f_idx in range(file_count):
                sf = exp.get_spheroid_file(f_idx)
                md = sf.get_metadata() or {}
                pos = md.get("peak_amplitude_positions")
                # treat None or NaN as missing
                if pos is None:
                    missing.append((r_idx, f_idx))
                else:
                    try:
                        import math
                        if isinstance(pos, float) and math.isnan(pos):
                            missing.append((r_idx, f_idx))
                    except Exception:
                        pass
        return missing

    def isComplete(self):
        # Page is complete only if EVERY file in EVERY replicate has peaks.
        return len(self._missing_peaks()) == 0
    
    #def isComplete(self):
        #group_analysis = self.wizard().group_analysis
        #try:
           #exp = group_analysis.get_single_experiments(self.current_rep_index)
            #actual_file_index = self.file_index_mapping[self.current_file_index]
            #sph_file = exp.get_spheroid_file(actual_file_index)
            #metadata = sph_file.get_metadata()
            #return metadata.get("peak_amplitude_positions") is not None
        #except Exception:
            #return False

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

        # Check file type
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        file_type = settings.value("file_type", "None", type=str)

        # Choose the appropriate processor
        #if file_type == "Spontaneous":
            #from core.processing.spontaneous_peak_detector import FindAmplitudeMultiple
            #processor = FindAmplitudeMultiple(peak_pos)
        #else:
            #processor = FindAmplitude(peak_pos)

        #for exp in group_analysis.get_experiments():
            #exp.set_processing_steps([processor])
            #exp.run()

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

    def _set_peak_controls_enabled(self, enabled: bool):
        """
        
        """
        self.peak_slider.setEnabled(enabled)
        self.btn_adj_peak.setEnabled(enabled)
        if not enabled:
            # also clear any temporary selection so nothing is drawn
            self.temp_peak = None