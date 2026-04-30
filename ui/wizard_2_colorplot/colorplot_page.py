from PyQt5.QtWidgets import (
    QApplication, QComboBox, QWizardPage, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QProgressDialog, QSlider, QToolTip, QCheckBox, QListWidget, QSpinBox, QDialogButtonBox,
    QMessageBox, QGroupBox, QListWidgetItem, QDialog
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

import numpy as np
import os
import re
from core.flow_cell_experiment import FlowCellExperiment

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

        self.btn_filter = QPushButton("Filter Options")
        apply_custom_styles(self.btn_filter)
        self.btn_filter.clicked.connect(self.show_processing_options)
        # Buffer subtraction button – this is the ONLY entry point
        self.btn_buffer_sub = QPushButton("Buffer Subtraction…")
        apply_custom_styles(self.btn_buffer_sub)
        self.btn_buffer_sub.clicked.connect(self._open_buffer_sub_dialog)
        self.btn_save = QPushButton("Save Current Plots"); 
        apply_custom_styles(self.btn_save)
        self.btn_save.clicked.connect(self.save_IT_ColorPlot_Plots)
        self.btn_export = QPushButton("Export Current IT")
        apply_custom_styles(self.btn_export)
        self.btn_export.clicked.connect(self.save_processed_data_IT)
        self.btn_export_all = QPushButton("Export All ITs")
        apply_custom_styles(self.btn_export_all)
        self.btn_export_all.clicked.connect(self.save_all_ITs)
        # self.btn_adj_peak = QPushButton("Apply Peak Adjustment")
        # apply_custom_styles(self.btn_adj_peak)
        # self.btn_adj_peak.clicked.connect(self.adjust_peak_position)
        for b in (self.btn_prev, self.btn_next, self.btn_eval, self.btn_filter,
          self.btn_save, self.btn_export, self.btn_export_all):
            b.setAutoDefault(False)
            b.setDefault(False)
        file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)

        self.chk_peak_click = None
        if file_type == "Spontaneous":
            self.chk_peak_click = QCheckBox("Click to edit peaks")
            self.chk_peak_click.setToolTip("Left-click: add/make active; Right-click: remove nearest")
            self.chk_peak_click.toggled.connect(self._enable_peak_click_mode)

        self.btn_edit_peaks = QPushButton("Edit Peaks…")
        apply_custom_styles(self.btn_edit_peaks)
        self.btn_edit_peaks.clicked.connect(self.on_edit_peaks_clicked)

        # Layout for left panel
        left = QVBoxLayout()
        left.addWidget(self.btn_revert)
        left.addWidget(self.cbo_rep)
        left.addWidget(self.cbo_file)

        nav = QHBoxLayout(); nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next)
        left.addLayout(nav)
        left.addWidget(self.btn_filter)

        # Buffer subtraction button - always add to layout, visibility controlled in initializePage
        left.addWidget(self.btn_buffer_sub)
        self.btn_buffer_sub.setVisible(False)  # Hidden by default, shown for Flow Cell in initializePage

        left.addWidget(self.btn_eval)
        left.addWidget(self.btn_save)
        left.addWidget(self.btn_export)
        left.addWidget(self.btn_export_all)
        left.addWidget(self.chk_peak_click)
        left.addWidget(self.btn_edit_peaks)

        self.left_layout = left


        # Right plots
        self.main_plot = PlotCanvas(self, width=5, height=4)

        self.it_plot = PlotCanvas(self, width=4, height=3)
        self.cv_plot = None

        print(f"DEBUG: File type: {file_type}")
        if file_type == "Spontaneous":
            self.cv_plot = PlotCanvas(self, width=4, height=3)

        bottom = QHBoxLayout()
        bottom.addWidget(self.it_plot)
        if file_type == "Spontaneous":
            bottom.addWidget(self.cv_plot)
        self.bottom_layout = bottom

        right = QVBoxLayout()
        right.addWidget(self.main_plot)
        right.addLayout(bottom)

        # Main layout for the page
        main_layout = QVBoxLayout()
        content_layout = QHBoxLayout()
        content_layout.addLayout(left)
        content_layout.addLayout(right)
        # Make left panel ~1/3 and right panel ~2/3 of the available width
        content_layout.setStretch(0, 1)
        content_layout.setStretch(1, 2)
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
        def_index = 0

        group_analysis = self.wizard().group_analysis
        display_names_list = self.wizard().display_names_list
        
        # Show/hide buffer subtraction button based on file type
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        file_type = settings.value("file_type", "None", type=str)
        self.btn_buffer_sub.setVisible(file_type == "Flow Cell")
        print(f"[DEBUG] initializePage: file_type={file_type}, buffer_sub button visible={file_type == 'Flow Cell'}")
        
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
        # self._set_peak_controls_enabled(self.isComplete())

    def on_file_changed(self, index):
        """
        Handles file selection changes within a replicate.

        Args:
            index (int): Index of the newly selected file.
        """
        self.current_file_index = index
        self.update_file_display()
        # self._set_peak_controls_enabled(self.isComplete())

    def _enable_peak_click_mode(self, enabled: bool):
        """Attach/detach click-to-edit on BOTH colour and IT plots."""
        # Prepare canvas list (colour plot is mandatory; IT plot optional)
        canvases = []
        if hasattr(self, "main_plot") and hasattr(self.main_plot, "fig") and hasattr(self.main_plot.fig, "canvas"):
            canvases.append(self.main_plot.fig.canvas)
        if hasattr(self, "it_plot") and hasattr(self.it_plot, "fig") and hasattr(self.it_plot.fig, "canvas"):
            canvases.append(self.it_plot.fig.canvas)

        if not hasattr(self, "_mpl_cids"):
            self._mpl_cids = []

        # Disconnect any existing handlers first
        if self._mpl_cids:
            for (cv, cid) in list(self._mpl_cids):
                try:
                    cv.mpl_disconnect(cid)
                except Exception:
                    pass
            self._mpl_cids = []

        if enabled:
            for cv in canvases:
                try:
                    cid = cv.mpl_connect("button_press_event", self._on_plot_click)
                except Exception:
                    cid = None
                self._mpl_cids.append((cv, cid))

    def _on_plot_click(self, ev):
        """Generic handler: left-click add/make active; right-click remove-nearest.
        Converts x to sample index depending on which axes was clicked (IT = seconds).
        """
        if ev.inaxes is None or ev.xdata is None:
            return
        group_analysis = self.wizard().group_analysis
        exp = group_analysis.get_single_experiments(self.current_rep_index)
        actual = self.file_index_mapping[self.current_file_index] if (
            self.file_index_mapping and self.current_file_index < len(self.file_index_mapping)) else self.current_file_index
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

        # Convert x to sample index
        if is_it_axis:
            x_sec = max(0.0, float(ev.xdata))
            idx = int(round(x_sec * fs))
        else:
            idx = int(round(ev.xdata))

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
        if ev.button == 1:  # left add/make active
            peaks.append(idx)
            seen, uniq = set(), []
            for p in peaks:
                if p not in seen:
                    uniq.append(p); seen.add(p)
            active = uniq.index(idx)
            meta_set_peaks_and_active(file_obj, uniq, active)
        elif ev.button == 3:  # right remove nearest
            if peaks:
                nearest = min(range(len(peaks)), key=lambda i: abs(peaks[i] - idx))
                del peaks[nearest]
                active = 0 if not peaks else max(0, min(active, len(peaks) - 1))
                meta_set_peaks_and_active(file_obj, peaks, active)

        self._refresh_plots_for_file(file_obj)

    def _clear_peak_overlays(self):
        # colour plot
        if hasattr(self, "_peak_lines_color"):
            for ln in self._peak_lines_color:
                try: ln.remove()
                except Exception: pass
        self._peak_lines_color = []
        # IT plot
        if hasattr(self, "_peak_lines_it"):
            for ln in self._peak_lines_it:
                try: ln.remove()
                except Exception: pass
        self._peak_lines_it = []

    def _draw_peak_overlays_color(self, file_obj):
        md = file_obj.get_metadata() or {}
        peaks, active = meta_get_peaks_and_active(md)
        ax = self.main_plot.fig.axes[0] if self.main_plot.fig.axes else self.main_plot.fig.add_subplot(111)
        # remove old lines
        if not hasattr(self, "_peak_lines_color"):
            self._peak_lines_color = []
        else:
            for ln in self._peak_lines_color:
                try: ln.remove()
                except Exception: pass
            self._peak_lines_color = []
        # draw
        for i, p in enumerate(peaks):
            ln = ax.axvline(x=p,
                            color="white" if i == active else "red",
                            linewidth=(2.5 if i == active else 1.5),
                            linestyle=("--" if i == active else ":"),
                            alpha=(0.9 if i == active else 0.5))
            self._peak_lines_color.append(ln)
        self.main_plot.fig.canvas.draw_idle()

    def _draw_peak_overlays_it(self, file_obj):
        md = file_obj.get_metadata() or {}
        peaks, active = meta_get_peaks_and_active(md)
        ax = self.it_plot.fig.axes[0] if self.it_plot.fig.axes else self.it_plot.fig.add_subplot(111)
        # remove old lines
        if not hasattr(self, "_peak_lines_it"):
            self._peak_lines_it = []
        else:
            for ln in self._peak_lines_it:
                try: ln.remove()
                except Exception: pass
            self._peak_lines_it = []
        # draw with seconds on x-axis
        fs = md.get('acquisition_frequency', 1)
        print("aq_freq")
        print(fs)
        try:
            fs = float(fs)
        except Exception:
            fs = 1.0
        fs = max(fs, 1.0)
        for i, p in enumerate(peaks):
            print("{{{{{{{")
            print(p)
            print("____")
            print(peaks)

            x_sec = float(p) / fs
            ln = ax.axvline(x=x_sec,
                            color="black" if i == active else "red",
                            linewidth=(2.5 if i == active else 1.0),
                            linestyle=("--" if i == active else ":"),
                            alpha=(0.9 if i == active else 0.5))
            self._peak_lines_it.append(ln)
        self.it_plot.fig.canvas.draw_idle()

    def _refresh_plots_for_file(self, file_obj):
        """Replot Colour, IT and, if available, CV using current metadata, then redraw overlays."""
        try:
            processed = file_obj.get_processed_data()
            metadata = file_obj.get_metadata() or {}
            peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position")

            # Replot Colour and IT
            self.main_plot.plot_color(processed_data=processed, peak_pos=peak_pos)
            self.it_plot.plot_IT(processed_data=processed, metadata=metadata,
                                 peak_position=peak_pos, temp_peak_detection=self.temp_peak,
                                 filepath=file_obj.get_filepath() if hasattr(file_obj, 'get_filepath') else None)

            # Replot CV if spontaneous mode and canvas exists
            file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)
            if file_type == "Spontaneous" and hasattr(self, "cv_plot") and self.cv_plot is not None:
                self.cv_plot.plot_cv(processed_data=processed, metadata=metadata,
                                     title_suffix=f"File {self.current_file_index + 1}")

            # Now overlays
            self._redraw_all_peak_overlays(file_obj)
        except Exception as e:
            print(f"ERROR: _refresh_plots_for_file failed: {e}")

    def _redraw_all_peak_overlays(self, file_obj):
        self._draw_peak_overlays_color(file_obj)
        self._draw_peak_overlays_it(file_obj)

    def on_edit_peaks_clicked(self):
        group_analysis = self.wizard().group_analysis
        exp = group_analysis.get_single_experiments(self.current_rep_index)
        actual = self.file_index_mapping[self.current_file_index] if (
                    self.file_index_mapping and self.current_file_index < len(self.file_index_mapping)) \
            else self.current_file_index
        file_obj = exp.get_spheroid_file(actual)

        md = file_obj.get_metadata() or {}
        peaks, active = meta_get_peaks_and_active(md)
        processed = file_obj.get_processed_data()
        max_index = int(processed.shape[0] - 1) if processed is not None else 0

        canvases = []
        if hasattr(self.main_plot, 'fig') and hasattr(self.main_plot.fig, 'canvas'):
            canvases.append(self.main_plot.fig.canvas)
        if hasattr(self.it_plot, 'fig') and hasattr(self.it_plot.fig, 'canvas'):
            canvases.append(self.it_plot.fig.canvas)

        file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)
        dlg = PeakEditorDialog(peaks, active_idx=active, max_index=max_index, canvases=canvases, file_type=file_type, parent=self)

        def _on_peaks_changed(new_peaks, new_active):
            meta_set_peaks_and_active(file_obj, new_peaks, new_active)
            self._refresh_plots_for_file(file_obj)
        try:
            dlg.peaks_changed.connect(_on_peaks_changed)
        except Exception:
            # Older dialog without signal; ignore
            pass

        if dlg.exec_() == QDialog.Accepted:
            try:
                new_peaks, new_active = dlg.result()
                meta_set_peaks_and_active(file_obj, new_peaks, new_active)
            except Exception:
                pass
            self._refresh_plots_for_file(file_obj)

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
                                 temp_peak_detection=self.temp_peak,
                                 filepath=sph_file.get_filepath() if hasattr(sph_file, 'get_filepath') else None)
            self._redraw_all_peak_overlays(sph_file)

            file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)
            if file_type == "Spontaneous":
                # Ensure CV plot exists and is attached once
                if not hasattr(self, "cv_plot") or self.cv_plot is None:
                    print("VISITED")
                    print(self.cv_plot)
                    self.cv_plot = PlotCanvas(self, width=4, height=3)
                    try:
                        self.bottom_layout.addWidget(self.cv_plot)
                    except Exception:
                        pass
                # Plot/update CV for the current file
                self.cv_plot.plot_cv(processed_data=processed_data, metadata=metadata,
                                      title_suffix=f"File {self.current_file_index + 1}")
                if not hasattr(self, "chk_peak_click") or not self.chk_peak_click:
                    self.chk_peak_click = QCheckBox("Click to edit peaks")
                    self.chk_peak_click.setToolTip("Left-click: add/make active; Right-click: remove nearest")
                    self.chk_peak_click.toggled.connect(self._enable_peak_click_mode)
                    self.left_layout.addWidget(self.chk_peak_click)
            elif file_type == "Stimulation":
                # Remove CV plot if present
                if hasattr(self, "cv_plot") and self.cv_plot is not None:
                    try:
                        self.bottom_layout.removeWidget(self.cv_plot)
                    except Exception:
                        try:
                            self.layout().removeWidget(self.cv_plot)
                        except Exception:
                            pass
                    self.cv_plot.setParent(None)
                    self.cv_plot.deleteLater()
                    self.cv_plot = None

                if hasattr(self, "chk_peak_click") and self.chk_peak_click:
                    try:
                        self.left_layout.removeWidget(self.chk_peak_click)
                    except Exception:
                        try:
                            self.layout().removeWidget(self.chk_peak_click)
                        except Exception:
                            pass
                    self.chk_peak_click.setParent(None)
                    self.chk_peak_click.deleteLater()
                    self.chk_peak_click = None

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

    def run_processing(self):
        """
        Runs the selected processing pipeline on the current experiment group.

        Automatically adds `FindAmplitude` as a mandatory step.
        Displays a progress dialog while processing.
        Updates visualizations after processing is complete.
        """
        # Revert to baseline before processing, but preserve buffer subtraction for Flow Cell
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        file_type = settings.value("file_type", "None", type=str)
        preserve_buffer = (file_type == "Flow Cell")  # Preserve buffer subtraction for Flow Cell
        
        print(f"[DEBUG] run_processing: file_type={file_type}, preserve_buffer_subtraction={preserve_buffer}")
        self.revert_processing(preserve_buffer_subtraction=preserve_buffer)
        
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

        # # Choose the appropriate amplitude finder based on file type
        # if file_type == "Spontaneous":
        #     mandatory = FindAmplitudeMultiple(peak_pos)
        # else:
        #     print("Using default amplitude finder______________")
        #     mandatory = FindAmplitude(peak_pos)

        # Keep all user processors EXCEPT any existing amplitude finders
        user_processors = self.selected_processors or []
        # processors = [p for p in user_processors if not isinstance(p, (FindAmplitude))]
        # Add the mandatory amplitude finder
        # processors.append(mandatory)
        print("DEBUG: User processors:")
        print(user_processors)

        group_analysis.set_processing_options_exp(user_processors)
        for exp in group_analysis.get_experiments():
            exp.run()

        # Check for processing warnings and display them
        self._show_processing_warnings(group_analysis)

        self.update_file_display()
        self.completeChanged.emit()
        # self._set_peak_controls_enabled(self.isComplete())
        progress.close()


    def revert_processing(self, preserve_buffer_subtraction=False):
        """
        Revert processing for all experiments.
        
        Args:
            preserve_buffer_subtraction (bool): If True, preserve buffer subtraction.
                                                If False (default for manual button click),
                                                clear everything including buffer subtraction.
        """
        print(f"[DEBUG] ColorPlotPage.revert_processing: preserve_buffer_subtraction={preserve_buffer_subtraction}")
        group_analysis = self.wizard().group_analysis
        
        # If not preserving buffer subtraction, clear it completely
        if not preserve_buffer_subtraction:
            for exp in group_analysis.get_experiments():
                if hasattr(exp, 'clear_all_buffer_subtractions'):
                    exp.clear_all_buffer_subtractions()
                else:
                    exp.revert_processing(preserve_buffer_subtraction=False)
        else:
            # Just revert processing, preserving buffer subtraction
            for exp in group_analysis.get_experiments():
                exp.revert_processing(preserve_buffer_subtraction=True)
        
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

    def _show_processing_warnings(self, group_analysis):
        """Check for processing warnings and display them in a message box."""
        all_warnings = []
        for exp in group_analysis.get_experiments():
            for sf in exp.files:
                metadata = sf.get_metadata() or {}
                warnings = metadata.get('processing_warnings', [])
                all_warnings.extend(warnings)
                # Clear warnings after collecting
                if 'processing_warnings' in metadata:
                    del metadata['processing_warnings']
        
        if all_warnings:
            # Remove duplicates while preserving order
            unique_warnings = list(dict.fromkeys(all_warnings))
            warning_text = "\n\n".join(unique_warnings)
            QMessageBox.warning(
                self,
                "Processing Warnings",
                f"The following issues were encountered during processing:\n\n{warning_text}"
            )

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

    def _on_subtract_buffers_clicked(self):
        """
        Deprecated method - buffer subtraction now happens via the dialog.
        This should not be called anymore since the old widgets were removed.
        """
        QMessageBox.information(
            self, "Use Buffer Subtraction Button",
            "Please use the 'Buffer Subtraction...' button to perform buffer subtraction."
        )

    def _open_buffer_sub_dialog(self):
        """Open the buffer subtraction dialog only when user clicks the button."""
        print("[UI DEBUG] _open_buffer_sub_dialog called")
        
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        file_type = settings.value("file_type", "None", type=str)
        print(f"[UI DEBUG] Current file_type from settings: {file_type}")
        
        if file_type != "Flow Cell":
            print("[UI DEBUG] Not a Flow Cell experiment, showing info dialog")
            QMessageBox.information(
                self, "Not Flow Cell",
                "Buffer subtraction is only available for Flow Cell experiments."
            )
            return

        print("[UI DEBUG] Opening BufferSubtractionDialog...")
        dlg = BufferSubtractionDialog(self, self.wizard().group_analysis)
        result = dlg.exec_()
        print(f"[UI DEBUG] Dialog result: {'Accepted' if result == QDialog.Accepted else 'Rejected'}")
        
        if result == QDialog.Accepted:
            if hasattr(dlg, "selected_exp_index"):
                print(f"[UI DEBUG] Dialog has selected_exp_index: {dlg.selected_exp_index}, current_rep_index: {self.current_rep_index}")
                if dlg.selected_exp_index == self.current_rep_index:
                    print("[UI DEBUG] Updating file display...")
                    self.update_file_display()
            else:
                print("[UI DEBUG] Dialog does not have selected_exp_index attribute")


class BufferSubtractionDialog(QDialog):
    """
    Modal dialog to perform buffer subtraction on Flow Cell experiments.
    Lets user choose experiment, buffer files, and target files,
    then calls FlowCellExperiment.run_buffer_subtraction().
    """
    def __init__(self, parent, group_analysis):
        super().__init__(parent)
        self.setWindowTitle("Buffer Subtraction")
        self.group_analysis = group_analysis

        layout = QVBoxLayout(self)

        # Experiment selector
        self.cbo_exp = QComboBox()
        layout.addWidget(QLabel("Experiment:"))
        layout.addWidget(self.cbo_exp)

        # Buffer list
        self.lst_buffers = QListWidget()
        self.lst_buffers.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(QLabel("Buffer files:"))
        layout.addWidget(self.lst_buffers)

        # Targets list
        self.lst_targets = QListWidget()
        self.lst_targets.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(QLabel("Target files:"))
        layout.addWidget(self.lst_targets)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)

        # init experiments list
        names = parent.wizard().display_names_list
        self.cbo_exp.addItems(names)
        self.cbo_exp.currentIndexChanged.connect(self._refresh_lists)
        if names:
            self._refresh_lists(0)

    def _refresh_lists(self, exp_index: int):
        from core.flow_cell_experiment import FlowCellExperiment

        print(f"[UI DEBUG] _refresh_lists called for experiment index: {exp_index}")
        
        try:
            exp = self.group_analysis.get_single_experiments(exp_index)
            print(f"[UI DEBUG] Got experiment: {type(exp).__name__}")
        except Exception as e:
            print(f"[UI DEBUG] ERROR: Failed to get experiment: {e}")
            return

        self.lst_buffers.clear()
        self.lst_targets.clear()

        if not isinstance(exp, FlowCellExperiment):
            print(f"[UI DEBUG] Not a FlowCellExperiment (type: {type(exp).__name__}), lists cleared")
            return

        print(f"[UI DEBUG] Buffer indices: {exp.buffer_indices}")
        print(f"[UI DEBUG] Total files: {exp.get_file_count()}")
        
        # Buffers
        for idx in exp.buffer_indices:
            sf = exp.get_spheroid_file(idx)
            try:
                fname = os.path.basename(sf.get_filepath())
            except Exception:
                fname = f"File {idx}"
            print(f"[UI DEBUG] Adding buffer: {fname} (index {idx})")
            item = QListWidgetItem(fname)
            item.setBackground(Qt.lightGray)
            item.setData(Qt.UserRole, idx)
            self.lst_buffers.addItem(item)

        # Non‑buffers
        non_buf = [i for i in range(exp.get_file_count()) if i not in exp.buffer_indices]
        print(f"[UI DEBUG] Non-buffer indices: {non_buf}")
        
        for idx in non_buf:
            sf = exp.get_spheroid_file(idx)
            try:
                fname = os.path.basename(sf.get_filepath())
            except Exception:
                fname = f"File {idx}"
            print(f"[UI DEBUG] Adding target: {fname} (index {idx})")
            item = QListWidgetItem(fname)
            item.setData(Qt.UserRole, idx)
            self.lst_targets.addItem(item)
        
        print(f"[UI DEBUG] Refresh complete: {self.lst_buffers.count()} buffers, {self.lst_targets.count()} targets")

    def _on_ok(self):
        from core.flow_cell_experiment import FlowCellExperiment

        exp_index = self.cbo_exp.currentIndex()
        print(f"\n[UI DEBUG] Buffer subtraction dialog OK clicked, experiment index: {exp_index}")
        
        try:
            exp = self.group_analysis.get_single_experiments(exp_index)
            print(f"[UI DEBUG] Retrieved experiment: {type(exp).__name__}")
        except Exception as e:
            print(f"[UI DEBUG] ERROR: Failed to get experiment: {e}")
            QMessageBox.warning(self, "No experiment", "Unable to access selected experiment.")
            return

        if not isinstance(exp, FlowCellExperiment):
            print(f"[UI DEBUG] ERROR: Experiment is not FlowCellExperiment, it's {type(exp).__name__}")
            QMessageBox.warning(self, "Wrong type", "Selected experiment is not a Flow Cell experiment.")
            return

        buf_indices = [it.data(Qt.UserRole) for it in self.lst_buffers.selectedItems()]
        print(f"[UI DEBUG] Selected buffer indices: {buf_indices}")
        
        if not buf_indices:
            print("[UI DEBUG] ERROR: No buffer files selected")
            QMessageBox.warning(self, "No buffers selected", "Please select at least one buffer file.")
            return

        tgt_items = self.lst_targets.selectedItems()
        tgt_indices = [it.data(Qt.UserRole) for it in tgt_items]
        if not tgt_indices:
            tgt_indices = [i for i in range(exp.get_file_count()) if i not in exp.buffer_indices]
            print(f"[UI DEBUG] No targets selected, using all non-buffer files: {tgt_indices}")
        else:
            print(f"[UI DEBUG] Selected target indices: {tgt_indices}")

        msg = (f"Subtract mean of {len(buf_indices)} buffer file(s)\n"
               f"from {len(tgt_indices)} target file(s) in experiment {exp_index + 1}?")
        reply = QMessageBox.question(self, "Confirm Buffer Subtraction", msg,
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            print("[UI DEBUG] User cancelled buffer subtraction")
            return

        print("[UI DEBUG] Calling exp.run_buffer_subtraction()...")
        # For Flow Cell: use processed_data to allow filtering before buffer subtraction
        ok = exp.run_buffer_subtraction(
            buffer_indices=buf_indices,
            target_indices=tgt_indices,
            use_processed_for_buffers=True,  # Use processed data for buffers
            write_to_processed=True,
            use_processed_as_source=True,  # Subtract from processed data of targets
        )
        
        print(f"[UI DEBUG] Buffer subtraction returned: {ok}")
        
        if not ok:
            print("[UI DEBUG] ERROR: Buffer subtraction failed")
            QMessageBox.warning(self, "Subtraction failed",
                                "Could not compute mean buffer or subtract; check data shapes.")
            return

        print("[UI DEBUG] Buffer subtraction successful!")
        QMessageBox.information(self, "Success", 
                               f"Buffer subtraction completed successfully!\n"
                               f"{len(tgt_indices)} files were processed.")
        
        self.selected_exp_index = exp_index
        self.accept()
