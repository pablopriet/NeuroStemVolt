from PyQt5.QtWidgets import (
    QApplication, QComboBox, QWizardPage, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QProgressDialog, QSlider, QToolTip,
QCheckBox, QListWidget, QSpinBox, QDialogButtonBox, QMessageBox
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

        left.addWidget(self.chk_peak_click)
        left.addWidget(self.btn_edit_peaks)

        self.left_layout = left


        # Right plots
        self.main_plot = PlotCanvas(self, width=5, height=4)

        self.it_plot = PlotCanvas(self, width=4, height=3)
        self.cv_plot = None

        print(f"DEBUG: File type: {file_type}")
        if file_type == "Spontaneous":
            # Add CV plot canvas
            self.cv_plot = PlotCanvas(self, width=4, height=3)

        bottom = QHBoxLayout()
        bottom.addWidget(self.it_plot)
        if file_type == "Spontaneous":
            bottom.addWidget(self.cv_plot)  # Add the CV plot here
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

        # Determine if click came from the IT axes (seconds) or colour plot (seconds)
        it_ax = None
        color_ax = None
        try:
            it_ax = self.it_plot.fig.axes[0] if self.it_plot and self.it_plot.fig.axes else None
        except Exception:
            it_ax = None
        try:
            color_ax = self.main_plot.fig.axes[0] if self.main_plot and self.main_plot.fig.axes else None
        except Exception:
            color_ax = None
        
        is_it_axis = (it_ax is not None and ev.inaxes is it_ax)
        is_color_axis = (color_ax is not None and ev.inaxes is color_ax)

        # Sampling frequency for seconds→samples conversion
        # Use same logic as plot_IT and overlays: prefer metadata, fall back to QSettings
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        if 'acquisition_frequency' in md:
            try:
                fs = float(md['acquisition_frequency'])
            except Exception:
                fs = settings.value("acquisition_frequency", 10, type=int)
        else:
            fs = settings.value("acquisition_frequency", 10, type=int)
        fs = max(float(fs), 1.0)
        
        print(f"Click handler: fs={fs}, ev.xdata={ev.xdata}")

        # Convert x to sample index
        # Both IT plot and color plot now use seconds on x-axis
        if is_it_axis or is_color_axis:
            x_sec = max(0.0, float(ev.xdata))
            idx = int(round(x_sec * fs))
            print(f"Converted click: x_sec={x_sec:.3f} -> idx={idx}")
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
        # draw with seconds on x-axis - use same freq logic as plot_IT
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        if 'acquisition_frequency' in md:
            try:
                fs = float(md['acquisition_frequency'])
            except Exception:
                fs = settings.value("acquisition_frequency", 10, type=int)
        else:
            fs = settings.value("acquisition_frequency", 10, type=int)
        fs = max(float(fs), 1.0)
        
        for i, p in enumerate(peaks):
            # Convert sample index to seconds
            x_sec = float(p) / fs
            ln = ax.axvline(x=x_sec,
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
        # draw with seconds on x-axis - use same freq logic as plot_IT
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        if 'acquisition_frequency' in md:
            try:
                fs = float(md['acquisition_frequency'])
            except Exception:
                fs = settings.value("acquisition_frequency", 10, type=int)
        else:
            fs = settings.value("acquisition_frequency", 10, type=int)
        fs = max(float(fs), 1.0)
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
            self.main_plot.plot_color(processed_data=processed, peak_pos=peak_pos, metadata=metadata)
            self.it_plot.plot_IT(processed_data=processed, metadata=metadata,
                                 peak_position=peak_pos, temp_peak_detection=self.temp_peak)

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

        # Get acquisition frequency for time conversion
        acq_freq = md.get('acquisition_frequency', 10)
        try:
            acq_freq = float(acq_freq)
        except Exception:
            acq_freq = 10.0

        file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)
        dlg = PeakEditorDialog(peaks, active_idx=active, max_index=max_index, canvases=canvases, 
                               file_type=file_type, acq_freq=acq_freq, parent=self)

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

            self.main_plot.plot_color(processed_data=processed_data, peak_pos=peak_pos, metadata=metadata)
            self.it_plot.plot_IT(processed_data=processed_data, metadata=metadata, peak_position=peak_pos,
                                 temp_peak_detection=self.temp_peak)
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

        # # Choose the appropriate amplitude finder based on file type
        # if file_type == "Spontaneous":
        #     mandatory = FindAmplitudeMultiple(peak_pos)
        # else:
        #     print("Using default amplitude finder______________")
        #     mandatory = FindAmplitude(peak_pos)

        # Keep all user processors EXCEPT any existing amplitude finders
        user_processors = self.selected_processors or []
        processors = [p for p in user_processors if not isinstance(p, (FindAmplitude, FindAmplitudeMultiple))]

        # Insert a FindAmplitude BEFORE Normalize so peak values are in context
        has_normalize = any(isinstance(p, Normalize) for p in processors)
        if has_normalize:
            # Build new list with FindAmplitude inserted before Normalize
            reordered = []
            for p in processors:
                if isinstance(p, Normalize):
                    # Insert a pre-normalization amplitude finder
                    if file_type == "Spontaneous":
                        reordered.append(FindAmplitudeMultiple(peak_pos))
                    else:
                        reordered.append(FindAmplitude(peak_pos))
                reordered.append(p)
            processors = reordered

        # Add the mandatory amplitude finder at the end (runs on normalized data)
        processors.append(mandatory)
        print(processors)

        group_analysis.set_processing_options_exp(user_processors)
        for exp in group_analysis.get_experiments():
            exp.run()

        # Check for processing warnings and display them
        self._show_processing_warnings(group_analysis)

        self.update_file_display()
        self.completeChanged.emit()
        # self._set_peak_controls_enabled(self.isComplete())
        progress.close()


    def revert_processing(self):
        group_analysis = self.wizard().group_analysis
        for exp in group_analysis.get_experiments():
            exp.revert_processing()
        # disarm the slider & clear temp point
        # self._set_peak_controls_enabled(False)
        self.update_file_display()
        self.completeChanged.emit()

    def show_processing_options(self):
        """
        Opens the processing options dialog and updates selected processors.

        Retrieves the user’s choices and instantiates corresponding Processor objects.
        When Normalize is selected, FindAmplitude is automatically inserted before it
        to provide the normalization factor, then FindAmplitude runs again after Normalize.
        """
        dlg = ProcessingOptionsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            selected_names = dlg.get_selected_processors()
            peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position", type=int)
            
            # Build processor list, inserting FindAmplitude before Normalize if needed
            processors = []
            normalize_enabled = "Normalize" in selected_names
            
            for name in selected_names:
                # If Normalize is enabled, insert FindAmplitude right before it
                if name == "Normalize" and normalize_enabled:
                    # Add FindAmplitude first pass (for normalization factor)
                    processors.append(dlg.get_processor_instance("Find Amplitude", peak_pos))
                
                proc = dlg.get_processor_instance(name, peak_pos)
                if proc is not None:
                    processors.append(proc)
            
            # Always add Find Amplitude at the end (on potentially normalized data)
            # This is the "real" amplitude finding pass
            find_amp = dlg.get_processor_instance("Find Amplitude", peak_pos)
            if find_amp is not None:
                processors.append(find_amp)
            
            self.selected_processors = processors

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

    # def _set_peak_controls_enabled(self, enabled: bool):
    #     """
    #
    #     """
    #     self.peak_slider.setEnabled(enabled)
    #     self.btn_adj_peak.setEnabled(enabled)
    #     if not enabled:
    #         # also clear any temporary selection so nothing is drawn
    #         self.temp_peak = None