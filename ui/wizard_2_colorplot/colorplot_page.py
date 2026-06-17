from PyQt5.QtWidgets import (
    QApplication, QComboBox, QWizardPage, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QDialog, QProgressDialog, QSlider, QToolTip, QCheckBox, QMessageBox, QDoubleSpinBox,
    QTextEdit, QDialogButtonBox, QFileDialog,
)
from PyQt5.QtCore import QSettings, Qt, QEvent

import os
from core.output_manager import OutputManager
from ui.utils.styles import apply_custom_styles
from ui.widgets.plot_canvas import PlotCanvas
from ui.wizard_2_colorplot.processing_dialog import ProcessingOptionsDialog
from ui.wizard_2_colorplot.peak_editor import PeakEditorDialog
from ui.utils.peaks import meta_get_peaks_and_active, meta_set_peaks_and_active

from core.output_manager import OutputManager
from core.processing import *
# Import both amplitude processors at module level
from core.processing.find_amplitude import FindAmplitude
from core.processing.spontaneous_peak_detector import FindAmplitudeMultiple
from core.processing.drift_correction import DriftCorrection

import json
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
        self._last_processing_warnings = []

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

        self.btn_prev = QPushButton("◀");
        apply_custom_styles(self.btn_prev)
        self.btn_prev.setToolTip("Previous file in the current replicate")
        self.btn_next = QPushButton("▶")
        apply_custom_styles(self.btn_next)
        self.btn_next.setToolTip("Next file in the current replicate")
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
        self.btn_export_cv = QPushButton("Export CV Plot")
        apply_custom_styles(self.btn_export_cv)
        self.btn_export_cv.clicked.connect(self.save_cv_plot)

        self.btn_export_all_cv = QPushButton("Export All CVs")
        apply_custom_styles(self.btn_export_all_cv)
        self.btn_export_all_cv.setToolTip("Export CV curve for every file as a CSV")
        self.btn_export_all_cv.clicked.connect(self.save_all_cv_csv)

        self.btn_session_export = QPushButton("Export Session")
        apply_custom_styles(self.btn_session_export)
        self.btn_session_export.setToolTip("Save peaks and filter settings to a JSON file")
        self.btn_session_export.clicked.connect(self.export_session)

        self.btn_session_import = QPushButton("Import Session")
        apply_custom_styles(self.btn_session_import)
        self.btn_session_import.setToolTip("Restore peaks and filter settings from a JSON file")
        self.btn_session_import.clicked.connect(self.import_session)

        for b in (self.btn_prev, self.btn_next, self.btn_eval, self.btn_filter,
          self.btn_save, self.btn_export, self.btn_export_all, self.btn_export_cv,
          self.btn_export_all_cv, self.btn_session_export, self.btn_session_import):
            b.setAutoDefault(False)
            b.setDefault(False)

        file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)

        self.btn_adj_peak = None

        self.btn_peak_click = None
        self.btn_peak_click = QCheckBox("Edit on Plot")
        self.btn_peak_click.setCheckable(True)
        self.btn_peak_click.setToolTip("Left-click: add/make active; Right-click: remove nearest")
        self.btn_peak_click.toggled.connect(self._enable_peak_click_mode)
        self.btn_peak_click.toggled.connect(self._update_peak_click_style)
        apply_custom_styles(self.btn_peak_click)


        self.btn_edit_peaks = QPushButton("Edit Peaks")
        apply_custom_styles(self.btn_edit_peaks)
        self.btn_edit_peaks.clicked.connect(self.on_edit_peaks_clicked)

        # Replicate navigation buttons
        self.btn_rep_prev = QPushButton("◀")
        apply_custom_styles(self.btn_rep_prev)
        self.btn_rep_prev.setToolTip("Previous replicate")
        self.btn_rep_prev.clicked.connect(self.on_rep_prev_clicked)
        self.btn_rep_next = QPushButton("▶")
        apply_custom_styles(self.btn_rep_next)
        self.btn_rep_next.setToolTip("Next replicate")
        self.btn_rep_next.clicked.connect(self.on_rep_next_clicked)
        for b in (self.btn_rep_prev, self.btn_rep_next):
            b.setAutoDefault(False); b.setDefault(False)

        # Standalone normalize toggle
        self.btn_normalize = QPushButton("Normalize")
        apply_custom_styles(self.btn_normalize)
        self.btn_normalize.setCheckable(True)
        self.btn_normalize.setToolTip("Toggle normalization and re-run")
        self.btn_normalize.clicked.connect(self._toggle_normalize)
        self.btn_normalize.setAutoDefault(False); self.btn_normalize.setDefault(False)

        # Active filters info label
        self.lbl_filters = QLabel("Active: None")
        self.lbl_filters.setStyleSheet("color: palette(windowtext); font-size: 9pt; font-style: italic; padding: 2px 0px;")
        self.lbl_filters.setWordWrap(True)

        # Peak detection report button — hidden until warnings exist
        self.btn_report = QPushButton("⚠ Peak Report")
        self.btn_report.setStyleSheet(
            "background-color: #e67e22; color: white; font-weight: bold;"
            " border-radius: 10px; padding: 4px 8px; font-size: 9pt;")
        self.btn_report.setAutoDefault(False)
        self.btn_report.setDefault(False)
        self.btn_report.hide()
        self.btn_report.clicked.connect(self._show_peak_report)

        def _sec(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "font-weight: bold; font-size: 9pt;"
                " padding-top: 6px; border-top: 1px solid palette(mid);")
            return lbl

        left = QVBoxLayout()
        left.setSpacing(4)

        # ── Navigation ───────────────────────────────
        left.addWidget(_sec("Navigation"))
        left.addWidget(self.cbo_rep)
        rep_nav = QHBoxLayout()
        rep_nav.addWidget(self.btn_rep_prev)
        rep_nav.addWidget(self.btn_rep_next)
        left.addLayout(rep_nav)
        left.addWidget(self.cbo_file)
        file_nav = QHBoxLayout()
        file_nav.addWidget(self.btn_prev)
        file_nav.addWidget(self.btn_next)
        left.addLayout(file_nav)

        # ── Processing ───────────────────────────────
        left.addWidget(_sec("Processing"))
        proc_row = QHBoxLayout()
        proc_row.addWidget(self.btn_filter)
        proc_row.addWidget(self.btn_eval)
        left.addLayout(proc_row)
        norm_row = QHBoxLayout()
        norm_row.addWidget(self.btn_normalize)
        norm_row.addWidget(self.btn_revert)
        left.addLayout(norm_row)
        left.addWidget(self.lbl_filters)
        left.addWidget(self.btn_report)
        left.addWidget(_sec("Peak Editing"))
        peak_edit_row = QHBoxLayout()
        peak_edit_row.addWidget(self.btn_peak_click)
        peak_edit_row.addWidget(self.btn_edit_peaks)
        left.addLayout(peak_edit_row)

        # ── Export ───────────────────────────────────
        left.addWidget(_sec("Export"))
        left.addWidget(self.btn_save)
        export_row = QHBoxLayout()
        export_row.addWidget(self.btn_export)
        export_row.addWidget(self.btn_export_all)
        left.addLayout(export_row)
        cv_export_row = QHBoxLayout()
        cv_export_row.addWidget(self.btn_export_cv)
        cv_export_row.addWidget(self.btn_export_all_cv)
        left.addLayout(cv_export_row)
        session_row = QHBoxLayout()
        session_row.addWidget(self.btn_session_export)
        session_row.addWidget(self.btn_session_import)
        left.addLayout(session_row)

        left.addStretch()

        # ── Visualization ────────────────────────────
        self.chk_show_cv = QCheckBox("Show CV Plot")
        self.chk_show_cv.toggled.connect(self._on_show_cv_toggled)

        self.chk_cv_ref = QCheckBox("Ref line")
        self.chk_cv_ref.setEnabled(False)
        self.chk_cv_ref.toggled.connect(self._redraw_cv)

        self.spin_cv_ref = QDoubleSpinBox()
        self.spin_cv_ref.setRange(-1.5, 2.0)
        self.spin_cv_ref.setSingleStep(0.05)
        self.spin_cv_ref.setValue(0.0)
        self.spin_cv_ref.setDecimals(2)
        self.spin_cv_ref.setSuffix(" V")
        self.spin_cv_ref.setEnabled(False)
        self.spin_cv_ref.valueChanged.connect(self._redraw_cv)

        # Colorbar upper-limit control. 0.0 nA stays pinned at the same point of the
        # colormap (the 3:2 / 0.6 logic); raising this just stretches the color range.
        self.chk_color_limit = QCheckBox("Manual color limit")
        self.chk_color_limit.setToolTip(
            "Set the upper end of the colorbar (nA). The lower end is fixed at "
            "-2/3 of the top (e.g. top 1.0 -> bottom -0.67, top 14 -> bottom -9.33), "
            "which keeps 0.0 nA at the same color and makes reduction (negative) "
            "peaks stand out. Applies to both the on-screen plot and the export.")
        self.chk_color_limit.toggled.connect(self._on_color_limit_changed)

        self.spin_color_limit = QDoubleSpinBox()
        self.spin_color_limit.setRange(0.05, 100.0)
        self.spin_color_limit.setSingleStep(0.1)
        self.spin_color_limit.setValue(1.0)
        self.spin_color_limit.setDecimals(2)
        self.spin_color_limit.setSuffix(" nA")
        self.spin_color_limit.setEnabled(False)
        self.spin_color_limit.valueChanged.connect(self._on_color_limit_changed)

        left.addWidget(_sec("Visualization"))
        left.addWidget(self.chk_show_cv)
        cv_ref_row = QHBoxLayout()
        cv_ref_row.addWidget(self.chk_cv_ref)
        cv_ref_row.addWidget(self.spin_cv_ref)
        left.addLayout(cv_ref_row)
        color_limit_row = QHBoxLayout()
        color_limit_row.addWidget(self.chk_color_limit)
        color_limit_row.addWidget(self.spin_color_limit)
        left.addLayout(color_limit_row)

        self.left_layout = left

        # Right plots
        self.main_plot = PlotCanvas(self, width=5, height=4)
        self.it_plot = PlotCanvas(self, width=4, height=3)
        self.cv_plot = PlotCanvas(self, width=4, height=3)
        self.cv_plot.setVisible(False)

        bottom = QHBoxLayout()
        bottom.addWidget(self.it_plot)
        self.bottom_layout = bottom

        right = QVBoxLayout()
        right.addWidget(self.main_plot)
        right.addLayout(bottom)
        right.addWidget(self.cv_plot)   # CV gets its own full-width row below IT

        # Main layout for the page
        main_layout = QVBoxLayout()
        content_layout = QHBoxLayout()
        content_layout.addLayout(left)
        content_layout.addLayout(right)
        content_layout.setStretch(0, 1)
        content_layout.setStretch(1, 2)
        main_layout.addLayout(content_layout)

        # Footer
        footer = QLabel("© 2025 Hashemi Lab · NeuroStemVolt · v1.0.0")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("""
            color: palette(windowtext);
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

        self._peak_lines_color = []
        self._peak_lines_it = []

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

            self.main_plot.plot_color(processed_data=processed_data, peak_pos=peak_pos,
                                      vmax=self._current_vmax())

            self.it_plot.plot_IT(processed_data=processed_data, metadata=metadata, peak_position=peak_pos,
                                 temp_peak_detection=self.temp_peak)

            self._redraw_all_peak_overlays(sph_file)

            if self.chk_show_cv.isChecked():
                ref_voltage = self.spin_cv_ref.value() if self.chk_cv_ref.isChecked() else None
                self.cv_plot.plot_cv(processed_data=processed_data, metadata=metadata,
                                     title_suffix=f"File {self.current_file_index + 1}",
                                     ref_voltage=ref_voltage)
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

    def run_processing(self):
        """
        Runs the selected processing pipeline on the current experiment group.

        Automatically adds `FindAmplitude` as a mandatory step.
        Displays a progress dialog while processing.
        Updates visualizations after processing is complete.
        """
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

        # Choose the appropriate amplitude finder (uses dialog params from QSettings)
        mandatory = self._make_peak_detector(peak_pos)

        # Strip any stray peak detectors — mandatory is appended at the end
        user_processors = self.selected_processors or []
        processors = [p for p in user_processors
                      if not isinstance(p, (FindAmplitude, FindAmplitudeMultiple))]

        # Normalize needs a FindAmplitude pass for its scale factor
        has_normalize = any(isinstance(p, Normalize) for p in processors)
        if has_normalize:
            reordered = []
            for p in processors:
                if isinstance(p, Normalize):
                    reordered.append(self._make_peak_detector(peak_pos))
                reordered.append(p)
            processors = reordered

        # FindAmplitude always runs last
        processors.append(mandatory)

        group_analysis.set_processing_options_exp(processors)
        for exp in group_analysis.get_experiments():
            exp.run()

        # Check for processing warnings and display them
        self._show_processing_warnings(group_analysis)

        self.update_file_display()
        self.completeChanged.emit()
        self._set_peak_controls_enabled(self.isComplete())
        self._update_filters_label()
        progress.close()


    def _make_peak_detector(self, peak_pos):
        """
        Instantiate the appropriate peak detector using parameters saved by the
        processing dialog (stored in QSettings under 'processing_params').
        Falls back to class defaults if no saved params exist.
        """
        import json
        settings  = QSettings("HashemiLab", "NeuroStemVolt")
        file_type = settings.value("file_type", "None", type=str)

        saved_raw   = settings.value("processing_params", type=str)
        saved_params = json.loads(saved_raw) if saved_raw else {}
        fa_params   = saved_params.get("Find Amplitude", None)

        if file_type == "Multi-Peak":
            prominence_fraction = 0.05
            max_peaks  = 10
            min_dist   = 0.5
            if isinstance(fa_params, list) and len(fa_params) >= 3:
                try: prominence_fraction = float(fa_params[0]) / 100.0
                except ValueError: pass
                try: max_peaks = int(fa_params[1])
                except ValueError: pass
                try: min_dist = float(fa_params[2])
                except ValueError: pass
            return FindAmplitudeMultiple(peak_pos,
                                         prominence_fraction=prominence_fraction,
                                         max_peaks=max_peaks,
                                         min_peak_distance_sec=min_dist)
        else:
            prominence_fraction = 0.10
            min_height_na = 0.03
            if isinstance(fa_params, list) and len(fa_params) >= 2:
                try: prominence_fraction = float(fa_params[0]) / 100.0
                except ValueError: pass
                try: min_height_na = float(fa_params[1])
                except ValueError: pass
            return FindAmplitude(peak_pos,
                                  prominence_fraction=prominence_fraction,
                                  min_height_na=min_height_na)

    def revert_processing(self):
        group_analysis = self.wizard().group_analysis
        for exp in group_analysis.get_experiments():
            exp.revert_processing()
        self.selected_processors = []
        self._set_peak_controls_enabled(False)
        self.update_file_display()
        self.completeChanged.emit()
        # Reset normalize button, filters label, and evaluate button
        self.btn_normalize.setChecked(False)
        self.btn_normalize.setText("Normalize")
        apply_custom_styles(self.btn_normalize)
        self.lbl_filters.setText("Active: None")
        self.btn_eval.setText("Evaluate")
        # Clear peak report
        self._last_processing_warnings = []
        self.btn_report.hide()
        self.btn_report.setText("⚠ Peak Report")

    def _build_processors_from_dialog(self, dlg):
        """Instantiate processors from dialog selections, preserving normalize button state."""
        selected_names = dlg.get_selected_processors()
        peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position", type=int)
        processors = []
        for name in selected_names:
            proc = dlg.get_processor_instance(name, peak_pos)
            if proc is not None:
                processors.append(proc)
        # Preserve normalize if the standalone button is active
        if self.btn_normalize.isChecked() and not any(isinstance(p, Normalize) for p in processors):
            processors.append(Normalize())
        self.selected_processors = processors

    def show_processing_options(self):
        """Opens the filter options dialog. Apply runs filters only; OK just closes."""
        dlg = ProcessingOptionsDialog(self)

        def _on_apply():
            self._build_processors_from_dialog(dlg)
            self._run_filters_only()
            self.btn_eval.setText("Find Peaks")

        dlg.apply_requested.connect(_on_apply)
        dlg.revert_requested.connect(self.revert_processing)

        if dlg.exec_() == QDialog.Accepted:
            # OK was clicked — update processor list (no auto-run)
            self._build_processors_from_dialog(dlg)

    def _run_filters_only(self):
        """Run selected filter processors without peak detection, preserving edited peaks."""
        group_analysis = self.wizard().group_analysis
        peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position", type=int)

        progress = QProgressDialog("Applying filters…", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        # Save peak positions before filtering so we can restore them after
        saved_peaks = {}
        for exp in group_analysis.get_experiments():
            for i in range(exp.get_file_count()):
                sf = exp.get_spheroid_file(i)
                md = sf.get_metadata() or {}
                pos = md.get("peak_amplitude_positions")
                active = md.get("active_peak_index") or 0
                saved_peaks[(id(exp), i)] = (pos, active)

        # Exclude peak detectors — all other processors (including drift) run here
        user_processors = self.selected_processors or []
        processors = [p for p in user_processors
                      if not isinstance(p, (FindAmplitude, FindAmplitudeMultiple))]

        # Normalize needs a FindAmplitude pass for its scale factor
        has_normalize = any(isinstance(p, Normalize) for p in processors)
        if has_normalize:
            reordered = []
            for p in processors:
                if isinstance(p, Normalize):
                    reordered.append(self._make_peak_detector(peak_pos))
                reordered.append(p)
            processors = reordered

        group_analysis.set_processing_options_exp(processors)
        for exp in group_analysis.get_experiments():
            exp.run()

        # Restore peak positions — recompute values from the newly filtered data
        for exp in group_analysis.get_experiments():
            for i in range(exp.get_file_count()):
                sf = exp.get_spheroid_file(i)
                pos, active = saved_peaks.get((id(exp), i), (None, 0))
                if pos is not None:
                    if isinstance(pos, (int, float)):
                        peaks = [int(pos)]
                    else:
                        peaks = [int(p) for p in list(pos) if p is not None]
                    if peaks:
                        meta_set_peaks_and_active(sf, peaks, active or 0)

        self._show_processing_warnings(group_analysis)
        self.update_file_display()
        self._update_filters_label()
        self.completeChanged.emit()
        self._set_peak_controls_enabled(self.isComplete())
        progress.close()

    def _show_processing_warnings(self, group_analysis):
        """Collect warnings from all files and update the Peak Report button."""
        all_warnings = []
        for exp in group_analysis.get_experiments():
            for sf in exp.files:
                metadata = sf.get_metadata() or {}
                warnings = metadata.get('processing_warnings', [])
                if warnings:
                    fname = os.path.basename(sf.get_filepath())
                    for w in warnings:
                        all_warnings.append(f"[{fname}]\n{w}")
                if 'processing_warnings' in metadata:
                    del metadata['processing_warnings']

        self._last_processing_warnings = all_warnings

        if all_warnings:
            n = len(all_warnings)
            self.btn_report.setText(f"⚠ Peak Report ({n})")
            self.btn_report.show()
        else:
            self.btn_report.hide()
            self.btn_report.setText("⚠ Peak Report")

    def _show_peak_report(self):
        """Open a scrollable dialog showing the last peak detection report."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Peak Detection Report")
        dlg.resize(520, 400)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("font-family: monospace; font-size: 10pt;")

        if self._last_processing_warnings:
            text.setPlainText("\n\n".join(self._last_processing_warnings))
        else:
            text.setPlainText("No issues detected.")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Issues encountered during the last processing run:"))
        layout.addWidget(text)
        layout.addWidget(buttons)

        dlg.exec_()

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

    def on_rep_prev_clicked(self):
        """Navigate to the previous replicate."""
        idx = self.cbo_rep.currentIndex()
        if idx > 0:
            self.cbo_rep.setCurrentIndex(idx - 1)

    def on_rep_next_clicked(self):
        """Navigate to the next replicate."""
        idx = self.cbo_rep.currentIndex()
        if idx < self.cbo_rep.count() - 1:
            self.cbo_rep.setCurrentIndex(idx + 1)

    def _toggle_normalize(self, checked: bool):
        """Toggle normalization and re-filter, preserving edited peaks."""
        if checked:
            self.btn_normalize.setText("✓ Normalize")
            self.btn_normalize.setStyleSheet(
                "background-color: #27ae60; color: white; font-weight: bold;"
                " border-radius: 12px; padding: 6px;")
            if not any(isinstance(p, Normalize) for p in self.selected_processors):
                self.selected_processors.append(Normalize())
        else:
            self.btn_normalize.setText("Normalize")
            apply_custom_styles(self.btn_normalize)
            self.selected_processors = [p for p in self.selected_processors
                                        if not isinstance(p, Normalize)]
        self._run_filters_only()

    def _update_filters_label(self):
        """Update the 'Active filters' label to reflect current selected_processors."""
        name_map = {
            'Normalize': 'Normalize',
            'BaselineCorrection': 'Baseline',
            'SmoothData': 'Smooth',
            'StimArtifactRemoval': 'Artifact Removal',
            'InvertData': 'Invert',
        }
        names = [name_map.get(type(p).__name__, type(p).__name__)
                 for p in self.selected_processors
                 if not isinstance(p, (FindAmplitude, FindAmplitudeMultiple))]
        self.lbl_filters.setText(f"Active: {', '.join(names)}" if names else "Active: None")

    def _enable_peak_click_mode(self, enabled: bool):
        canvases = []
        if hasattr(self, "main_plot") and hasattr(self.main_plot, "fig"):
            canvases.append(self.main_plot.fig.canvas)
        if hasattr(self, "it_plot") and hasattr(self.it_plot, "fig"):
            canvases.append(self.it_plot.fig.canvas)
        if not hasattr(self, "_mpl_cids"):
            self._mpl_cids = []
        for (cv, cid) in list(self._mpl_cids):
            try: cv.mpl_disconnect(cid)
            except Exception: pass
        self._mpl_cids = []
        if enabled:
            for cv in canvases:
                try: cid = cv.mpl_connect("button_press_event", self._on_plot_click)
                except Exception: cid = None
                self._mpl_cids.append((cv, cid))

    def _update_peak_click_style(self, checked: bool):
        if checked:
            self.btn_peak_click.setText("Edit on Plot ✎")
            self.btn_peak_click.setStyleSheet(
                "background-color: #e67e22; color: white; font-weight: bold; border-radius: 12px; padding: 6px;")
        else:
            self.btn_peak_click.setText("Edit on Plot")
            apply_custom_styles(self.btn_peak_click)

    def _on_plot_click(self, ev):
        """Generic handler: left-click add/make active; right-click remove-nearest.
        Converts x to sample index depending on which axes was clicked (IT = seconds).
        """
        if ev.inaxes is None or ev.xdata is None:
            return
        group_analysis = self.wizard().group_analysis
        exp = group_analysis.get_single_experiments(self.current_rep_index)
        actual = self.file_index_mapping[self.current_file_index] if (
                self.file_index_mapping and self.current_file_index < len(
            self.file_index_mapping)) else self.current_file_index
        file_obj = exp.get_spheroid_file(actual)

        md = file_obj.get_metadata() or {}
        peaks, active = meta_get_peaks_and_active(md)

        # Determine if click came from the IT axes (seconds) or colour plot (samples)
        it_ax = None
        try:
            it_ax = self.it_plot.fig.axes[0] if self.it_plot and self.it_plot.fig.axes else None
        except Exception:
            it_ax = None
        is_it_axis = (it_ax is not None and ev.inaxes is it_ax)

        # Sampling frequency for IT seconds→samples conversion
        fs = md.get('acquisition_frequency', 1)
        try:
            fs = float(fs)
        except Exception:
            fs = 1.0
        fs = max(fs, 1.0)

        # Convert x to sample index — both the colour plot and the IT plot
        # use seconds on their x-axis, so always convert seconds → samples.
        x_sec = max(0.0, float(ev.xdata))
        idx = int(round(x_sec * fs))

        # Clamp to data length
        total_len = None
        try:
            it = getattr(file_obj, 'get_processed_data_IT', lambda: None)()
            if it is not None:
                total_len = len(it)
        except Exception:
            total_len = None
        if total_len is None:
            try:
                mat = getattr(file_obj, 'get_processed_data', lambda: None)()
                if mat is not None:
                    total_len = mat.shape[0]
            except Exception:
                total_len = None
        if total_len is None:
            total_len = idx + 1
        idx = max(0, min(idx, int(total_len) - 1))

        # Apply change
        file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)
        if ev.button == 1:  # left add/make active
            if file_type == "Single Peak":
                # Replace the existing peak rather than adding a second one
                uniq = [idx]
                active = 0
            else:
                peaks.append(idx)
                seen, uniq = set(), []
                for p in peaks:
                    if p not in seen:
                        uniq.append(p)
                        seen.add(p)
                active = uniq.index(idx)
            meta_set_peaks_and_active(file_obj, uniq, active)
        elif ev.button == 3:  # right remove nearest
            if peaks:
                nearest = min(range(len(peaks)), key=lambda i: abs(peaks[i] - idx))
                del peaks[nearest]
                active = 0 if not peaks else max(0, min(active, len(peaks) - 1))
                meta_set_peaks_and_active(file_obj, peaks, active)

        self._refresh_plots_for_file(file_obj)

    def _draw_peak_overlays_color(self, file_obj):
        md = file_obj.get_metadata() or {}
        peaks, active = meta_get_peaks_and_active(md)
        if not self.main_plot.fig.axes: return
        ax = self.main_plot.fig.axes[0]
        if not hasattr(self, "_peak_lines_color"): self._peak_lines_color = []
        for ln in self._peak_lines_color:
            try: ln.remove()
            except Exception: pass
        self._peak_lines_color = []
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        try: fs = max(float(md.get('acquisition_frequency', settings.value("acquisition_frequency", 10, type=int))), 1.0)
        except Exception: fs = 10.0
        for i, p in enumerate(peaks):
            ln = ax.axvline(x=float(p) / fs,   # convert sample index → seconds
                            color="white" if i == active else "red",
                            linewidth=2.5 if i == active else 1.5,
                            linestyle="--" if i == active else ":",
                            alpha=0.9 if i == active else 0.5)
            self._peak_lines_color.append(ln)
        self.main_plot.fig.canvas.draw_idle()

    def _draw_peak_overlays_it(self, file_obj):
        md = file_obj.get_metadata() or {}
        peaks, active = meta_get_peaks_and_active(md)
        if not self.it_plot.fig.axes: return
        ax = self.it_plot.fig.axes[0]
        if not hasattr(self, "_peak_lines_it"): self._peak_lines_it = []
        for ln in self._peak_lines_it:
            try: ln.remove()
            except Exception: pass
        self._peak_lines_it = []
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        try: fs = max(float(md.get('acquisition_frequency', settings.value("acquisition_frequency", 10, type=int))), 1.0)
        except Exception: fs = 10.0
        for i, p in enumerate(peaks):
            ln = ax.axvline(x=float(p)/fs,
                            color="black" if i == active else "red",
                            linewidth=2.5 if i == active else 1.0,
                            linestyle="--" if i == active else ":",
                            alpha=0.9 if i == active else 0.5)
            self._peak_lines_it.append(ln)
        self.it_plot.fig.canvas.draw_idle()

    def _current_vmax(self):
        """Manual upper color limit (nA) to pass to get_norm, or None for auto.

        The spin box sets the top of the colorbar directly; the lower limit is then
        fixed at -(2/3) of the top, so 0.0 nA stays at a constant ~0.4 up the colorbar.
        """
        if self.chk_color_limit.isChecked():
            return self.spin_color_limit.value()
        return None

    def _on_color_limit_changed(self, *args):
        """Persist the color-limit setting (so the export matches) and redraw."""
        manual = self.chk_color_limit.isChecked()
        self.spin_color_limit.setEnabled(manual)
        qs = QSettings("HashemiLab", "NeuroStemVolt")
        qs.setValue("color_vmax_manual", manual)
        qs.setValue("color_vmax", self._current_vmax() if manual else "")
        # Redraw the current file's color plot with the new limit, if one is loaded.
        try:
            exp = self.wizard().group_analysis.get_single_experiments(self.current_rep_index)
            actual = self.file_index_mapping[self.current_file_index] if (
                self.file_index_mapping and self.current_file_index < len(self.file_index_mapping)
            ) else self.current_file_index
            self._refresh_plots_for_file(exp.get_spheroid_file(actual))
        except Exception:
            pass  # no data loaded yet

    def _redraw_all_peak_overlays(self, file_obj):
        self._draw_peak_overlays_color(file_obj)
        self._draw_peak_overlays_it(file_obj)

    def _refresh_plots_for_file(self, file_obj):
        try:
            processed = file_obj.get_processed_data()
            metadata = file_obj.get_metadata() or {}
            peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position")
            self.main_plot.plot_color(processed_data=processed, peak_pos=peak_pos,
                                      vmax=self._current_vmax())
            self.it_plot.plot_IT(processed_data=processed, metadata=metadata,
                                 peak_position=peak_pos, temp_peak_detection=self.temp_peak)
            if self.chk_show_cv.isChecked():
                ref_voltage = self.spin_cv_ref.value() if self.chk_cv_ref.isChecked() else None
                self.cv_plot.plot_cv(processed_data=processed, metadata=metadata,
                                     title_suffix=f"File {self.current_file_index + 1}",
                                     ref_voltage=ref_voltage)
            self._redraw_all_peak_overlays(file_obj)
            # Re-evaluate Next button — manual peak changes count toward completion
            self.completeChanged.emit()
        except Exception as e:
            print(f"ERROR: _refresh_plots_for_file failed: {e}")

    def on_edit_peaks_clicked(self):
        group_analysis = self.wizard().group_analysis
        exp = group_analysis.get_single_experiments(self.current_rep_index)
        actual = self.file_index_mapping[self.current_file_index] if (
            self.file_index_mapping and self.current_file_index < len(self.file_index_mapping)) else self.current_file_index
        file_obj = exp.get_spheroid_file(actual)
        md = file_obj.get_metadata() or {}
        peaks, active = meta_get_peaks_and_active(md)
        processed = file_obj.get_processed_data()
        max_index = int(processed.shape[0] - 1) if processed is not None else 0
        canvases = []
        if hasattr(self.main_plot, 'fig'): canvases.append(self.main_plot.fig.canvas)
        if hasattr(self.it_plot, 'fig'): canvases.append(self.it_plot.fig.canvas)
        try: acq_freq = max(float(md.get('acquisition_frequency', 10)), 1.0)
        except Exception: acq_freq = 10.0
        file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)
        dlg = PeakEditorDialog(peaks, active_idx=active, max_index=max_index, canvases=canvases,
                               file_type=file_type, acq_freq=acq_freq, parent=self)
        def _on_peaks_changed(new_peaks, new_active):
            meta_set_peaks_and_active(file_obj, new_peaks, new_active)
            self._refresh_plots_for_file(file_obj)
        try: dlg.peaks_changed.connect(_on_peaks_changed)
        except Exception: pass
        if dlg.exec_() == QDialog.Accepted:
            try:
                new_peaks, new_active = dlg.result()
                meta_set_peaks_and_active(file_obj, new_peaks, new_active)
            except Exception: pass
            self._refresh_plots_for_file(file_obj)

    def _set_peak_controls_enabled(self, enabled: bool):
        if self.btn_adj_peak is not None:
            self.btn_adj_peak.setEnabled(enabled)
        if not enabled:
            self.temp_peak = None

    def save_all_ITs(self):
        """
        Saves all I-T profiles from all experiments to the specified output folder.
        """
        group_analysis = self.wizard().group_analysis
        output_folder_path = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        OutputManager.save_all_ITs(group_analysis, output_folder_path)

    def _current_spheroid_file(self):
        """Return the spheroid file currently shown, resolving the display index
        through file_index_mapping so exports match what's on screen."""
        exp = self.wizard().group_analysis.get_single_experiments(self.current_rep_index)
        actual = self.file_index_mapping[self.current_file_index] if (
            self.file_index_mapping and self.current_file_index < len(self.file_index_mapping)
        ) else self.current_file_index
        return exp.get_spheroid_file(actual)

    def save_IT_ColorPlot_Plots(self):
        """
        Saves the color plot and I-T profile visualizations for the current file.
        """
        sph_file = self._current_spheroid_file()
        output_folder_path = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        sph_file.visualize_color_plot_data(title_suffix="", save_path=output_folder_path,
                                           vmax=self._current_vmax())
        sph_file.visualize_IT_profile(QSettings("HashemiLab", "NeuroStemVolt").value("output_folder"))

    def save_processed_data_IT(self):
        """
        Saves the processed I-T data array (not a figure) for the current file.
        """
        sph_file = self._current_spheroid_file()
        output_folder_path = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        OutputManager.save_IT_profile(sph_file, output_folder_path)

    def _on_show_cv_toggled(self, checked: bool):
        """Show/hide the CV plot; populate it immediately when first made visible."""
        self.cv_plot.setVisible(checked)
        self.chk_cv_ref.setEnabled(checked)
        self.spin_cv_ref.setEnabled(checked and self.chk_cv_ref.isChecked())
        if not checked:
            return
        self._redraw_cv()

    def _redraw_cv(self):
        """Redraw the CV plot with current data and reference line settings."""
        if not self.chk_show_cv.isChecked():
            return
        # Enable/disable spin based on checkbox state
        self.spin_cv_ref.setEnabled(self.chk_cv_ref.isChecked())
        ref_voltage = self.spin_cv_ref.value() if self.chk_cv_ref.isChecked() else None
        try:
            group_analysis = self.wizard().group_analysis
            exp = group_analysis.get_single_experiments(self.current_rep_index)
            actual = (self.file_index_mapping[self.current_file_index]
                      if self.file_index_mapping and self.current_file_index < len(self.file_index_mapping)
                      else self.current_file_index)
            sph_file = exp.get_spheroid_file(actual)
            processed_data = sph_file.get_processed_data()
            metadata = sph_file.get_metadata()
            self.cv_plot.plot_cv(processed_data=processed_data, metadata=metadata,
                                 title_suffix=f"File {self.current_file_index + 1}",
                                 ref_voltage=ref_voltage)
        except Exception:
            pass  # no data loaded yet

    def save_all_cv_csv(self):
        """
        Export the CV curve (current vs voltage at the peak time point) for every
        file in every replicate as a single multi-indexed CSV file.

        Rows  = voltage steps (V)
        Columns = MultiIndex (Replicate, Filename)

        Files with no peak detected use the middle time point as fallback.
        Files with no processed data are filled with NaN.
        """
        import pandas as pd
        from core.spheroid_file import Waveforms

        output_folder = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        if not output_folder or not os.path.isdir(output_folder):
            QMessageBox.warning(self, "No Output Folder",
                                "Please set a valid output folder in Experiment Settings.")
            return

        group_analysis = self.wizard().group_analysis
        experiments    = group_analysis.get_experiments()
        if not experiments:
            return

        # Discover voltage axis from the first file that has processed data
        voltage    = None
        n_voltages = None
        for exp in experiments:
            for sf in exp.files:
                data = sf.get_processed_data()
                if data is not None:
                    n_voltages = data.shape[1]
                    try:
                        wf      = Waveforms(-0.5, [-0.7, 1.1], -0.5, 600, n_voltages)
                        voltage = wf.voltage_waveform()
                    except Exception:
                        voltage = np.linspace(-0.5, 1.1, n_voltages)
                    break
            if voltage is not None:
                break

        if voltage is None:
            QMessageBox.warning(self, "No Processed Data",
                                "No processed data found. Run Evaluate first.")
            return

        columns_tuples = []
        cv_columns     = []

        for exp_idx, exp in enumerate(experiments):
            rep_name = f"Rep{exp_idx + 1}"
            for sf in exp.files:
                fname = os.path.basename(sf.get_filepath())
                columns_tuples.append((rep_name, fname))

                data = sf.get_processed_data()
                if data is None:
                    cv_columns.append(np.full(len(voltage), np.nan))
                    continue

                md       = sf.get_metadata() or {}
                peak_pos = md.get("peak_amplitude_positions")

                if peak_pos is None:
                    time_point = data.shape[0] // 2
                elif isinstance(peak_pos, list):
                    active     = int(md.get("peak_amplitude_active_index", 0))
                    active     = max(0, min(active, len(peak_pos) - 1))
                    time_point = int(peak_pos[active])
                else:
                    try:
                        time_point = int(peak_pos)
                    except (TypeError, ValueError):
                        time_point = data.shape[0] // 2

                time_point = max(0, min(time_point, data.shape[0] - 1))
                cv_columns.append(data[time_point, :])

        columns = pd.MultiIndex.from_tuples(columns_tuples, names=["Replicate", "File"])
        df      = pd.DataFrame(
            np.column_stack(cv_columns) if cv_columns else np.empty((len(voltage), 0)),
            columns=columns,
        )
        df.index      = voltage
        df.index.name = "Voltage (V)"

        out_dir  = os.path.join(output_folder, "all_replicates_CVs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "All_CVs_all_replicates.csv")

        try:
            df.to_csv(out_path)
            QMessageBox.information(self, "CV Export Complete",
                                    f"All CVs exported to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed",
                                 f"Could not write CSV:\n{e}")

    def export_session(self):
        """
        Export peak selections and active filter settings to a JSON file.

        Saves:
          - peaks: per-file peak positions, active index and values (keyed by filename)
          - filters/pipeline: which processors were selected
          - filters/params: processor parameters
          - normalize_active: whether the Normalize toggle was on
        """
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Session", "", "NeuroStemVolt Session (*.nsv);;JSON Files (*.json)")
        if not path:
            return

        settings = QSettings("HashemiLab", "NeuroStemVolt")

        # ── Peak metadata ────────────────────────────────────────────────────
        peaks_data = {}
        group_analysis = self.wizard().group_analysis
        for exp in group_analysis.get_experiments():
            for sf in exp.files:
                md = sf.get_metadata() or {}
                positions = md.get("peak_amplitude_positions")
                if positions is None:
                    continue
                fname = os.path.basename(sf.get_filepath())
                # Normalise to list so JSON serialises cleanly
                if isinstance(positions, (list,)):
                    pos_list = [int(p) for p in positions]
                else:
                    try:
                        pos_list = [int(positions)]
                    except (TypeError, ValueError):
                        continue

                values = md.get("peak_amplitude_values")
                if isinstance(values, list):
                    val_list = [float(v) for v in values]
                elif values is not None:
                    try:
                        val_list = [float(values)]
                    except (TypeError, ValueError):
                        val_list = []
                else:
                    val_list = []

                active = int(md.get("peak_amplitude_active_index", 0))
                peaks_data[fname] = {
                    "positions":    pos_list,
                    "active_index": active,
                    "values":       val_list,
                }

        # ── Filter settings ──────────────────────────────────────────────────
        pipeline_raw = settings.value("processing_pipeline", type=str)
        params_raw   = settings.value("processing_params",   type=str)
        filters = {
            "pipeline":        json.loads(pipeline_raw) if pipeline_raw else [],
            "params":          json.loads(params_raw)   if params_raw   else {},
            "normalize_active": self.btn_normalize.isChecked(),
        }

        session = {"version": 1, "filters": filters, "peaks": peaks_data}

        try:
            with open(path, "w") as f:
                json.dump(session, f, indent=2)
            QMessageBox.information(
                self, "Session Exported",
                f"Session saved to:\n{path}\n\n"
                f"{len(peaks_data)} file(s) with peak data exported.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not write session file:\n{e}")

    def import_session(self):
        """
        Import peak selections and filter settings from a previously exported session file.

        - Peak metadata is restored to each matching file (matched by filename).
        - Filter pipeline and params are written back to QSettings and
          ``self.selected_processors`` is rebuilt so clicking Evaluate re-applies
          the same pipeline.
        - The display is refreshed and completeChanged is emitted so the Next
          button updates immediately.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Session", "", "NeuroStemVolt Session (*.nsv);;JSON Files (*.json)")
        if not path:
            return

        try:
            with open(path, "r") as f:
                session = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Could not read session file:\n{e}")
            return

        if session.get("version") != 1:
            QMessageBox.warning(self, "Import Warning",
                                "Session file version is unrecognised — attempting import anyway.")

        # ── Restore filter settings ──────────────────────────────────────────
        filters = session.get("filters", {})
        settings = QSettings("HashemiLab", "NeuroStemVolt")

        if "pipeline" in filters:
            settings.setValue("processing_pipeline", json.dumps(filters["pipeline"]))
        if "params" in filters:
            settings.setValue("processing_params", json.dumps(filters["params"]))

        # Rebuild selected_processors from the restored QSettings
        dlg = ProcessingOptionsDialog(self)
        self._build_processors_from_dialog(dlg)
        self._update_filters_label()

        normalize_active = filters.get("normalize_active", False)
        self.btn_normalize.setChecked(normalize_active)
        if normalize_active:
            self.btn_normalize.setText("✓ Normalize")
            self.btn_normalize.setStyleSheet(
                "background-color: #27ae60; color: white; font-weight: bold;"
                " border-radius: 12px; padding: 6px;")
        else:
            self.btn_normalize.setText("Normalize")
            apply_custom_styles(self.btn_normalize)

        # ── Restore peak metadata ────────────────────────────────────────────
        peaks_data = session.get("peaks", {})
        group_analysis = self.wizard().group_analysis
        matched = 0
        skipped = 0

        for exp in group_analysis.get_experiments():
            for sf in exp.files:
                fname = os.path.basename(sf.get_filepath())
                if fname not in peaks_data:
                    skipped += 1
                    continue
                entry    = peaks_data[fname]
                pos_list = entry.get("positions", [])
                active   = int(entry.get("active_index", 0))
                val_list = entry.get("values", [])

                md = sf.get_metadata()  # always returns the live dict

                # Write peak metadata back
                if len(pos_list) == 1:
                    md["peak_amplitude_positions"] = pos_list[0]
                    md["peak_amplitude_values"]    = val_list[0] if val_list else None
                else:
                    md["peak_amplitude_positions"] = pos_list
                    md["peak_amplitude_values"]    = val_list
                md["peak_amplitude_active_index"] = active
                matched += 1

        self.update_file_display()
        self.completeChanged.emit()
        self._set_peak_controls_enabled(self.isComplete())
        self.btn_eval.setText("Find Peaks")

        QMessageBox.information(
            self, "Session Imported",
            f"Peaks restored for {matched} file(s).\n"
            f"{skipped} file(s) had no saved peaks.\n\n"
            f"Filter settings have been restored. Re-apply settings in Filter Options.")
    def save_cv_plot(self):
        """
        Saves the current CV plot as a PNG to the output folder.
        """
        if not self.chk_show_cv.isChecked() or self.cv_plot is None:
            QMessageBox.warning(self, "CV Plot Not Visible",
                                "Enable the CV plot using 'Show CV Plot' before exporting.")
            return
        try:
            exp = self.wizard().group_analysis.get_single_experiments(self.current_rep_index)
            actual = self.file_index_mapping[self.current_file_index] if (
                self.file_index_mapping and self.current_file_index < len(self.file_index_mapping)
            ) else self.current_file_index
            sph_file = exp.get_spheroid_file(actual)
            base_name = os.path.splitext(os.path.basename(sph_file.get_filepath()))[0]
            output_folder_path = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
            output_folder_cv = os.path.join(output_folder_path, "CV_plots")
            os.makedirs(output_folder_cv, exist_ok=True)

            # Render into a fresh, larger off-screen figure rather than dumping the
            # small on-screen canvas, so the exported plot is high-resolution and
            # the labels/annotations aren't cramped.
            ref_voltage = self.spin_cv_ref.value() if self.chk_cv_ref.isChecked() else None
            export_canvas = PlotCanvas(parent=None, width=8, height=6, dpi=150)
            export_canvas.plot_cv(
                processed_data=sph_file.get_processed_data(),
                metadata=sph_file.get_metadata(),
                title_suffix=f"File {self.current_file_index + 1}",
                ref_voltage=ref_voltage,
            )

            save_path = os.path.join(output_folder_cv, f"{base_name}_CV.png")
            export_canvas.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            # Also export an SVG (vector) version for editing individual elements
            export_canvas.fig.savefig(os.path.join(output_folder_cv, f"{base_name}_CV.svg"),
                                      bbox_inches='tight')
            QMessageBox.information(self, "CV Plot Exported",
                                    f"CV plot saved to:\n{output_folder_cv}")
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", f"Could not save CV plot:\n{e}")
