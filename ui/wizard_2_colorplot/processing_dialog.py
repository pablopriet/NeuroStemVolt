from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QLineEdit, QDialogButtonBox, QWidget, QPushButton
)
from PyQt5.QtCore import QSettings, Qt, pyqtSignal
import json

from ui.utils.ui_helpers import make_labeled_field_with_help
from core.processing import BackgroundSubtraction, SavitzkyGolayFilter, RollingMean, GaussianSmoothing2D, \
    ButterworthFilter, BaselineCorrection, Normalize, FindAmplitude, ExponentialFitting, \
    StimArtifactRemoval, InvertData, DriftCorrection

from core.processing.spontaneous_peak_detector import FindAmplitudeMultiple

class ProcessingOptionsDialog(QDialog):
    apply_requested = pyqtSignal()   # emitted when Apply is clicked (dialog stays open)
    revert_requested = pyqtSignal()  # emitted when Reverse Changes is clicked
    """
    Dialog for configuring and selecting signal processing options.

    This UI allows users to:
    - Choose which preprocessing steps to apply.
    - Configure parameters (e.g., window sizes, smoothing regions).
    - Persist selections across sessions using QSettings.

    Attributes:
        qsettings (QSettings): Persistent storage for user preferences.
        processor_options (list): List of available processors and their default activation state.
        checkboxes (dict): Maps processor names to their associated QCheckBox.
        param_widgets (dict): Maps processor names to their parameter input widgets.
    """
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Filtering Options")
        self.qsettings = QSettings("HashemiLab", "NeuroStemVolt")

        self.processor_options = [
            ("Background Subtraction", True),
            ("Rolling Mean", False),
            ("Butterworth Filter", True),
            ("Savitzky-Golay Filter", False),
            ("Baseline Correction", True),
            ("Artifact Removal", True),
            ("Invert Data", False),
            # ("Drift Correction", False),
            ("Find Amplitude", True),
        ]

        self.checkboxes = {}
        self.param_widgets = {}
        layout = QVBoxLayout()

        saved = self.qsettings.value("processing_pipeline", type=str)
        saved_selection = json.loads(saved) if saved else []

        saved = self.qsettings.value("processing_params", type=str)
        saved_params = json.loads(saved) if saved else {}

        help_texts = {
            "Background Subtraction": "Subtracts baseline offset by averaging the signal between a specified 'start' and 'end' segment (given as data indices or time points at the beginning of the trace) and subtracting that mean from the entire recording.",
            "Rolling Mean": "Smooths the trace by computing a moving average over a sliding window of N points. The 'window size' parameter sets how many consecutive samples are included in each average. Larger windows yield smoother traces but can blur sharp features.",
            "Butterworth Filter": "Applies a low-pass filter while preserving waveform. The 'order' (p) controls the steepness of the filter roll-off, while 'cx' and 'cy' set the cutoff frequencies (Hz) in the time and voltage dimensions, respectively. Lower cx = more smoothing along the time axis.",
            "Savitzky-Golay Filter": "Fits a local polynomial of a given 'order' over each segment of the data to smooth noise. The 'window size' sets how many points are used per fit, while 'order' (the 'p' polynomial order) controls how closely the fit can follow rapid changes.",
            "Baseline Correction": "Prior to the onset of stimulation, if the file is not starting at 0.0 nA, Baseline Correction computes the mean current across the period prior to stimulation and subtracts this baseline from the entire signal.",
            "Artifact Removal": (
                "Detects and removes sudden large electrical noise spikes: sharp jumps in current that appear "
                "in only one or a few scans and stand out massively from the surrounding signal. "
                "Threshold controls how many times larger a jump must be (relative to the median) to be flagged. "
                "Pad sets how many neighbouring scans on each side of a detected region are also replaced. "
                "Max Scans sets the widest a single artefact is allowed to be, in scans. A spike usually produces "
                "two jumps, one as the signal leaves the baseline and one as it returns; if those two jumps are no "
                "more than Max Scans apart they are treated as one artefact and everything between them is "
                "interpolated, whereas if they are further apart they are treated as separate one-scan events. "
                "Leaving it blank uses twice the acquisition frequency, which is two seconds of recording. "
                "Note: this is NOT designed for stimulation artefacts. It is intended for random electrical "
                "interference that contaminates isolated scans. If no such noise is detected, the data is "
                "returned unchanged."
            ),
            "Invert Data": "Inverts the sign of the signal. Use this if your data is recorded with reversed polarity.",
            "Drift Correction": "Fits a linear trend to the baseline file amplitudes (files before treatment) and subtracts the extrapolated drift from all subsequent files. Requires at least 2 baseline files. Must run after peak detection.",
        }

        file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)

        for name, default_checked in self.processor_options:
            if name == "Find Amplitude":
                continue
            # Baseline Correction is not meaningful for Multi-Peak files
            if name == "Baseline Correction" and file_type == "Multi-Peak":
                continue

            # Create a vertical layout for each filter option
            filter_layout = QVBoxLayout()
            filter_layout.setSpacing(2)
            filter_layout.setContentsMargins(0, 0, 0, 0)

            cb = QCheckBox(name)
            cb.setChecked(name in saved_selection if saved_selection else default_checked)
            cb.setStyleSheet("font-weight: bold; font-size: 12px;")
            help_widget = make_labeled_field_with_help(name, cb, help_texts.get(name, "No help available."))
            filter_layout.addWidget(help_widget)
            self.checkboxes[name] = cb

            # Parameter widget (hidden by default)
            param_widget = None

            if name == "Background Subtraction":
                region_layout = QHBoxLayout()
                region_label = QLabel("Region (start, end) in seconds:")
                region_label.setStyleSheet("font-size: 11px; color: palette(windowtext); margin-left: 16px;")
                region_start = QLineEdit("0")
                region_end = QLineEdit("10")
                if "Background Subtraction" in saved_params:
                    start_str, end_str = saved_params["Background Subtraction"]
                    region_start.setText(start_str)
                    region_end.setText(end_str)
                region_layout.addWidget(region_label)
                region_layout.addWidget(region_start)
                region_layout.addWidget(region_end)
                region_container = QWidget()
                region_container.setLayout(region_layout)
                region_container.setContentsMargins(24, 0, 0, 0)  # Indent
                region_container.hide()
                param_widget = region_container
                self.param_widgets[name] = (region_start, region_end)
            elif name == "Savitzky-Golay Filter":
                sg_layout = QHBoxLayout()
                sg_label_w = QLabel("Window:")
                sg_label_w.setStyleSheet("font-size: 11px; color: palette(windowtext); margin-left: 16px;")
                sg_window = QLineEdit("5")
                sg_label_o = QLabel("Order:")
                sg_label_o.setStyleSheet("font-size: 11px; color: palette(windowtext);")
                sg_order = QLineEdit("2")
                if "Savitzky-Golay Filter" in saved_params:
                    w, p = saved_params["Savitzky-Golay Filter"]
                    sg_window.setText(w)
                    sg_order.setText(p)
                sg_layout.addWidget(sg_label_w)
                sg_layout.addWidget(sg_window)
                sg_layout.addWidget(sg_label_o)
                sg_layout.addWidget(sg_order)
                sg_container = QWidget()
                sg_container.setLayout(sg_layout)
                sg_container.setContentsMargins(24, 0, 0, 0)  # Indent
                sg_container.hide()
                param_widget = sg_container
                self.param_widgets[name] = (sg_window, sg_order)
            elif name == "Rolling Mean":
                rm_layout = QHBoxLayout()
                rm_label = QLabel("Window Size:")
                rm_label.setStyleSheet("font-size: 11px; color: palette(windowtext); margin-left: 16px;")
                rm_window = QLineEdit("5")
                if "Rolling Mean" in saved_params:
                    rm_window.setText(saved_params["Rolling Mean"])
                rm_layout.addWidget(rm_label)
                rm_layout.addWidget(rm_window)
                rm_container = QWidget()
                rm_container.setLayout(rm_layout)
                rm_container.setContentsMargins(24, 0, 0, 0)  # Indent
                rm_container.hide()
                param_widget = rm_container
                self.param_widgets[name] = rm_window
            elif name == "Butterworth Filter":
                bw_layout = QHBoxLayout()
                bw_label_p = QLabel("Order (p):")
                bw_label_p.setStyleSheet("font-size: 11px; color: palette(windowtext); margin-left: 16px;")
                bw_p = QLineEdit("4")
                bw_label_cx = QLabel("cx:")
                bw_label_cx.setStyleSheet("font-size: 11px; color: palette(windowtext);")
                bw_cx = QLineEdit("2.5")
                bw_label_cy = QLabel("cy:")
                bw_label_cy.setStyleSheet("font-size: 11px; color: palette(windowtext);")
                bw_cy = QLineEdit("37500.0")
                if "Butterworth Filter" in saved_params:
                    saved_bw = saved_params["Butterworth Filter"]
                    if isinstance(saved_bw, list) and len(saved_bw) >= 1:
                        bw_p.setText(saved_bw[0])
                    if isinstance(saved_bw, list) and len(saved_bw) >= 2:
                        bw_cx.setText(saved_bw[1])
                    if isinstance(saved_bw, list) and len(saved_bw) >= 3:
                        bw_cy.setText(saved_bw[2])
                bw_layout.addWidget(bw_label_p)
                bw_layout.addWidget(bw_p)
                bw_layout.addWidget(bw_label_cx)
                bw_layout.addWidget(bw_cx)
                bw_layout.addWidget(bw_label_cy)
                bw_layout.addWidget(bw_cy)
                bw_container = QWidget()
                bw_container.setLayout(bw_layout)
                bw_container.setContentsMargins(24, 0, 0, 0)  # Indent
                bw_container.hide()
                param_widget = bw_container
                self.param_widgets[name] = (bw_p, bw_cx, bw_cy)
            elif name == "Artifact Removal":
                ar_layout = QHBoxLayout()
                ar_label_t = QLabel("Threshold:")
                ar_label_t.setStyleSheet("font-size: 11px; color: palette(windowtext); margin-left: 16px;")
                ar_threshold = QLineEdit("8")
                ar_label_p = QLabel("Pad:")
                ar_label_p.setStyleSheet("font-size: 11px; color: palette(windowtext);")
                ar_pad = QLineEdit("2")
                ar_label_m = QLabel("Max Scans (blank=auto):")
                ar_label_m.setStyleSheet("font-size: 11px; color: palette(windowtext);")
                ar_max = QLineEdit("")
                if "Artifact Removal" in saved_params:
                    saved_ar = saved_params["Artifact Removal"]
                    if isinstance(saved_ar, list) and len(saved_ar) == 3:
                        ar_threshold.setText(saved_ar[0])
                        ar_pad.setText(saved_ar[1])
                        ar_max.setText(saved_ar[2])
                ar_layout.addWidget(ar_label_t)
                ar_layout.addWidget(ar_threshold)
                ar_layout.addWidget(ar_label_p)
                ar_layout.addWidget(ar_pad)
                ar_layout.addWidget(ar_label_m)
                ar_layout.addWidget(ar_max)
                ar_container = QWidget()
                ar_container.setLayout(ar_layout)
                ar_container.setContentsMargins(24, 0, 0, 0)
                ar_container.hide()
                param_widget = ar_container
                self.param_widgets[name] = (ar_threshold, ar_pad, ar_max)

            # Add parameter widget to filter layout if it exists
            if param_widget:
                filter_layout.addWidget(param_widget)

                # Show/hide parameter widget based on checkbox; resize dialog to fit
                def toggle_widget(checked, widget=param_widget):
                    widget.setVisible(checked)
                    self.adjustSize()
                cb.toggled.connect(toggle_widget)
                # Set initial visibility
                param_widget.setVisible(cb.isChecked())

            # Add the filter layout to the main dialog layout
            filter_container = QWidget()
            filter_container.setLayout(filter_layout)
            layout.addWidget(filter_container)

        # ── Peak detection parameters (always visible) ────────────────────────
        fa_saved = saved_params.get("Find Amplitude", None)

        if file_type == "Multi-Peak":
            layout.addWidget(self._build_multi_peak_params(fa_saved))
        else:
            fa_params_layout = QHBoxLayout()
            fa_params_layout.setContentsMargins(8, 2, 0, 0)
            fa_params_layout.setSpacing(6)

            fa_pct_lbl    = QLabel("Prominence (%):")
            fa_pct_lbl.setStyleSheet("font-size: 11px; color: palette(windowtext);")
            fa_pct        = QLineEdit("10")
            fa_pct.setFixedWidth(45)
            fa_height_lbl = QLabel("Min Height (nA):")
            fa_height_lbl.setStyleSheet("font-size: 11px; color: palette(windowtext);")
            fa_height     = QLineEdit("0.03")
            fa_height.setFixedWidth(55)
            if isinstance(fa_saved, list) and len(fa_saved) >= 2:
                fa_pct.setText(fa_saved[0])
                fa_height.setText(fa_saved[1])
            fa_params_layout.addWidget(fa_pct_lbl)
            fa_params_layout.addWidget(fa_pct)
            fa_params_layout.addWidget(fa_height_lbl)
            fa_params_layout.addWidget(fa_height)
            fa_params_layout.addStretch()
            self.param_widgets["Find Amplitude"] = (fa_pct, fa_height)

            fa_params_container = QWidget()
            fa_params_container.setLayout(fa_params_layout)
            layout.addWidget(fa_params_container)

        # Dialog buttons — OK / Apply / Reverse Changes / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        revert_btn = buttons.addButton("Reverse Changes", QDialogButtonBox.ResetRole)
        revert_btn.clicked.connect(self._on_revert)
        layout.addWidget(buttons)

        self.setLayout(layout)

    # ── Multi-Peak detector parameters ───────────────────────────────────────

    # Every FindAmplitudeMultiple parameter, as (key, label, default, width).
    # The first group is always visible; the second sits behind the "Advanced"
    # toggle so the common case stays uncluttered.
    MULTI_PEAK_CRITICAL = [
        ("prominence_pct",   "Prominence (%):",   "5",    45),
        ("min_height_na",    "Min Height (nA):",  "0.02", 55),
        ("max_peaks",        "Max Peaks:",        "10",   40),
        ("min_distance_sec", "Min Distance (s):", "0.5",  45),
    ]
    MULTI_PEAK_ADVANCED = [
        ("min_peak_width_scans", "Min Peak Width (scans):", "2",    45),
        ("min_rise_sec",         "Rise Window min (s):",    "0.2",  45),
        ("max_rise_sec",         "Rise Window max (s):",    "3.0",  45),
        ("min_decay_sec",        "Decay Window min (s):",   "1.0",  45),
        ("max_decay_sec",        "Decay Window max (s):",   "10.0", 45),
    ]

    MULTI_PEAK_HELP = (
        "Detection thresholds:\n"
        "• Prominence (%) — how far a peak must stand out from its surroundings, "
        "as a percentage of the signal's dynamic range (5–95th percentile).\n"
        "• Min Height (nA) — absolute current floor; anything below is ignored.\n"
        "• Max Peaks — only the N largest peaks in a file are kept.\n"
        "• Min Distance (s) — minimum gap between two accepted peaks.\n\n"
        "Advanced (shape validation): each candidate is checked for a rising edge and "
        "a decaying tail, searched adaptively inside these bounds.\n"
        "• Min Peak Width (scans) — rejects single-scan spikes.\n"
        "• Rise Window min/max (s) — a candidate whose rise is shorter than the minimum "
        "is rejected; the maximum is how far back the search looks.\n"
        "• Decay Window min/max (s) — same, for the decay after the peak.\n"
        "Widening the windows or lowering the minima accepts slower, broader events; "
        "tightening them keeps only sharp transients."
    )

    def _build_multi_peak_params(self, saved):
        """
        Build the FindAmplitudeMultiple parameter block.

        Critical thresholds are always visible; the shape-validation windows are
        collapsed behind an "Advanced" toggle. Every parameter of the detector
        is editable here.

        Args:
            saved (dict | list | None): previously saved "Find Amplitude" params.
                The legacy 3-item list form is still understood.

        Returns:
            QWidget: container holding the whole block.
        """
        # Legacy list form: [prominence_pct, max_peaks, min_distance_sec]
        if isinstance(saved, list):
            saved = dict(zip(("prominence_pct", "max_peaks", "min_distance_sec"), saved))
        elif not isinstance(saved, dict):
            saved = {}

        fields = {}

        def make_row(specs):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            for key, label, default, width in specs:
                lbl = QLabel(label)
                lbl.setStyleSheet("font-size: 11px; color: palette(windowtext);")
                edit = QLineEdit(str(saved.get(key, default)))
                edit.setFixedWidth(width)
                fields[key] = edit
                row.addWidget(lbl)
                row.addWidget(edit)
            row.addStretch()
            return row

        outer = QVBoxLayout()
        outer.setContentsMargins(8, 2, 0, 0)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Peak Detection")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        header.addWidget(make_labeled_field_with_help(
            "Peak Detection", title, self.MULTI_PEAK_HELP))
        header.addStretch()
        outer.addLayout(header)

        outer.addLayout(make_row(self.MULTI_PEAK_CRITICAL))

        # Advanced section — hidden until the user asks for it
        self.mp_advanced_btn = QPushButton("▸ Advanced")
        self.mp_advanced_btn.setCheckable(True)
        self.mp_advanced_btn.setFlat(True)
        self.mp_advanced_btn.setStyleSheet(
            "QPushButton { font-size: 11px; text-align: left; padding: 0px; border: none; }")
        self.mp_advanced_btn.setCursor(Qt.PointingHandCursor)

        advanced_box = QVBoxLayout()
        advanced_box.setContentsMargins(16, 0, 0, 0)
        advanced_box.setSpacing(2)
        advanced_box.addLayout(make_row(self.MULTI_PEAK_ADVANCED[:1]))
        advanced_box.addLayout(make_row(self.MULTI_PEAK_ADVANCED[1:3]))
        advanced_box.addLayout(make_row(self.MULTI_PEAK_ADVANCED[3:]))

        reset_btn = QPushButton("Reset Advanced to Defaults")
        reset_btn.setStyleSheet("font-size: 11px;")
        reset_btn.clicked.connect(self._reset_multi_peak_advanced)
        reset_row = QHBoxLayout()
        reset_row.setContentsMargins(0, 0, 0, 0)
        reset_row.addWidget(reset_btn)
        reset_row.addStretch()
        advanced_box.addLayout(reset_row)

        advanced_container = QWidget()
        advanced_container.setLayout(advanced_box)
        advanced_container.hide()

        def toggle_advanced(checked):
            advanced_container.setVisible(checked)
            self.mp_advanced_btn.setText("▾ Advanced" if checked else "▸ Advanced")
            self.adjustSize()
        self.mp_advanced_btn.toggled.connect(toggle_advanced)

        outer.addWidget(self.mp_advanced_btn)
        outer.addWidget(advanced_container)

        self.param_widgets["Find Amplitude"] = fields

        container = QWidget()
        container.setLayout(outer)
        return container

    def _reset_multi_peak_advanced(self):
        """Restore the advanced multi-peak validation fields to their defaults."""
        fields = self.param_widgets.get("Find Amplitude")
        if not isinstance(fields, dict):
            return
        for key, _label, default, _width in self.MULTI_PEAK_ADVANCED:
            if key in fields:
                fields[key].setText(default)

    def _on_apply(self):
        """Save current settings and emit apply_requested (dialog stays open)."""
        self._save_settings()
        self.apply_requested.emit()

    def _on_revert(self):
        """Emit revert_requested so the parent can roll back all processing."""
        self.revert_requested.emit()

    def _save_settings(self):
        """Persist the current processor selection and parameters to QSettings."""
        selected = self.get_selected_processors()
        self.qsettings.setValue("processing_pipeline", json.dumps(selected))
        out = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, dict):
                out[name] = {key: w.text() for key, w in widget.items()}
            elif isinstance(widget, tuple):
                out[name] = [w.text() for w in widget]
            else:
                out[name] = widget.text()
        self.qsettings.setValue("processing_params", json.dumps(out))

    def get_processor_instance(self, name, peak_position=None):
        """
        Instantiate the selected processor based on its name and parameters.

        Args:
            name (str): Name of the processing step.
            peak_position (int, optional): Voltage index of the peak, needed by some processors.

        Returns:
            Processor: An instance of a subclass of `Processor`, or None if not matched.
        """
        if name == "Background Subtraction":
            region_start, region_end = self.param_widgets[name]
            try:
                start = int(region_start.text())
                end = int(region_end.text())
            except ValueError:
                start, end = 0, 10
            return BackgroundSubtraction(region=(start, end))
        elif name == "Savitzky-Golay Filter":
            sg_window, sg_order = self.param_widgets[name]
            try:
                w = int(sg_window.text())
                p = int(sg_order.text())
            except ValueError:
                w, p = 20, 2
            return SavitzkyGolayFilter(w=w, p=p)
        elif name == "Rolling Mean":
            rm_window = self.param_widgets[name]
            try:
                window_size = int(rm_window.text())
            except ValueError:
                window_size = 5
            return RollingMean(window_size=window_size)
        elif name == "Gaussian Smoothing 2D":
            return GaussianSmoothing2D()
        elif name == "Butterworth Filter":
            bw_p, bw_cx, bw_cy = self.param_widgets[name]
            try:
                p = int(bw_p.text())
                cx = float(bw_cx.text())
                cy = float(bw_cy.text())
            except ValueError:
                p, cx, cy = 4, 2.5, 37500.0
            return ButterworthFilter(p=p, cx=cx, cy=cy)
        elif name == "Baseline Correction":
            return BaselineCorrection()
        elif name == "Normalize":
            return Normalize(peak_position)
        elif name == "Artifact Removal":
            if name in self.param_widgets:
                ar_threshold, ar_pad, ar_max = self.param_widgets[name]
                try:
                    threshold = float(ar_threshold.text())
                except ValueError:
                    threshold = 8
                try:
                    pad = int(ar_pad.text())
                except ValueError:
                    pad = 2
                max_scans = None
                if ar_max.text().strip():
                    try:
                        max_scans = int(ar_max.text())
                    except ValueError:
                        max_scans = None
                return StimArtifactRemoval(threshold=threshold, pad=pad, max_artifact_scans=max_scans)
            return StimArtifactRemoval()
        elif name == "Invert Data":
            return InvertData()
        elif name == "Drift Correction":
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            files_before_treatment = settings.value("files_before_treatment", 0, type=int)
            return DriftCorrection(files_before_treatment=files_before_treatment)
        elif name == "Find Amplitude":
            settings  = QSettings("HashemiLab", "NeuroStemVolt")
            file_type = settings.value("file_type", "None", type=str)
            fa_params = self.param_widgets.get("Find Amplitude")

            if file_type == "Multi-Peak":
                # fa_params maps parameter keys to their QLineEdits; hand the raw
                # text to the detector, which parses it and fills in defaults.
                texts = ({k: w.text() for k, w in fa_params.items()}
                         if isinstance(fa_params, dict) else fa_params)
                return FindAmplitudeMultiple.from_params(peak_position, texts)
            else:
                prominence_fraction = 0.10
                min_height_na = 0.03
                if isinstance(fa_params, tuple) and len(fa_params) >= 2:
                    try: prominence_fraction = float(fa_params[0].text()) / 100.0
                    except ValueError: pass
                    try: min_height_na = float(fa_params[1].text())
                    except ValueError: pass
                return FindAmplitude(peak_position,
                                     prominence_fraction=prominence_fraction,
                                     min_height_na=min_height_na)
        elif name == "Exponential Fitting":
            return ExponentialFitting()
        else:
            return None

    def get_selected_processors(self):
        """
        Retrieve a list of processor names selected by the user.

        Returns:
            list of str: Names of enabled processing steps.
        """
        return [name for name, cb in self.checkboxes.items() if cb.isChecked()]

    def accept(self):
        """Save settings and close the dialog."""
        self._save_settings()
        super().accept()