from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QLineEdit, QDialogButtonBox, QWidget, QPushButton
)
from PyQt5.QtCore import QSettings
import json

from ui.utils.styles import apply_custom_styles
from ui.utils.ui_helpers import make_labeled_field_with_help
from core.processing import InvertData, BackgroundSubtraction, SavitzkyGolayFilter, RollingMean, GaussianSmoothing2D, ButterworthFilter, BaselineCorrection, Normalize, FindAmplitude, ExponentialFitting, StimArtifactRemoval

from core.processing.spontaneous_peak_detector import FindAmplitudeMultiple

class ProcessingOptionsDialog(QDialog):
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
            ("Normalize", True),
            ("Invert Data", False),  # Now enabled, but unchecked by default
        ]

        file_type = QSettings("HashemiLab", "NeuroStemVolt").value("file_type", "None", type=str)
        self.file_type = file_type  # Store for later use
        self.checkboxes = {}
        self.param_widgets = {}
        layout = QVBoxLayout()

        saved = self.qsettings.value("processing_pipeline", type=str)
        saved_selection = json.loads(saved) if saved else []

        saved = self.qsettings.value("processing_params", type=str)
        saved_params = json.loads(saved) if saved else {}

        if file_type == "Spontaneous":
            self.processor_options.append(("Single Peak Detection", False))
            self.processor_options.append(("Multiple Peak Detection", True))
        else:
            self.processor_options.append(("Single Peak Detection", True))

        if file_type == "Stimulation":
            # Insert right after Background Subtraction (index 0) so it runs second in the pipeline
            self.processor_options.insert(1, ("Artifact Removal", True))

        help_texts = {
            "Background Subtraction": "Subtracts baseline offset by averaging the signal between a specified 'start' and 'end' segment (given as data indices or time points at the beginning of the trace) and subtracting that mean from the entire recording.",
            "Rolling Mean": "Smooths the trace by computing a moving average over a sliding window of N points. The 'window size' parameter sets how many consecutive samples are included in each average. Larger windows yield smoother traces but can blur sharp features.",
            "Butterworth Filter": "Applies a low-pass filter while preserving waveform. The 'order' (p) controls the steepness of the filter roll-off, while 'cx' and 'cy' set the cutoff frequencies (Hz) in the time and voltage dimensions, respectively.",
            "Savitzky-Golay Filter": "Fits a local polynomial of a given 'order' over each segment of the data to smooth noise. The 'window size' sets how many points are used per fit, while 'order' (the 'p' polynomial order) controls how closely the fit can follow rapid changes.",
            "Baseline Correction": "Removes baseline drift from the signal.",
            "Normalize": "Normalizes each trace based on the peak amplitude of the first file within each replicate.",
            "Invert Data": "Inverts the sign of all data points. Use if your data is 'upside down' and you need to flip it.",
            "Single Peak Detection": "Detects peaks throughout the signal using a simple thresholding algorithm. Useful for analyzing stimulation activity patterns.",
            "Multiple Peak Detection": "Detects multiple peaks throughout the signal using adaptive validation windows. Useful for analyzing spontaneous activity patterns.",
            "Artifact Removal": "Auto-detects and removes the stimulation artifact from files marked 'Has Artifact'. Finds large scan-to-scan jumps via a MAD-based threshold, then linearly interpolates across each detected region in every voltage column. Threshold: MAD multiplier (lower = more sensitive). Pad: extra scans buffered around each artifact edge. Max Scans: largest artifact window allowed (0 = auto, defaults to 2 × acquisition rate).",
        }

        for name, default_checked in self.processor_options:
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

            # Disable Single Peak Detection and Normalize for Spontaneous file type
            if self.file_type == "Spontaneous":
                if name == "Single Peak Detection":
                    cb.setEnabled(False)
                    cb.setChecked(False)
                elif name == "Normalize":
                    cb.setEnabled(False)
                    cb.setChecked(False)

            # Parameter widget (hidden by default)
            param_widget = None

            if name == "Background Subtraction":
                region_layout = QHBoxLayout()
                region_label = QLabel("Region (start, end) in seconds:")
                region_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
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
                sg_label_w.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                sg_window = QLineEdit("20")
                sg_label_o = QLabel("Order:")
                sg_label_o.setStyleSheet("font-size: 11px; color: #555;")
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
                rm_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
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
            elif name == "Artifact Removal":
                ar_layout = QVBoxLayout()

                threshold_layout = QHBoxLayout()
                threshold_label = QLabel("Threshold (MAD multiplier):")
                threshold_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                threshold_edit = QLineEdit("8")
                threshold_layout.addWidget(threshold_label)
                threshold_layout.addWidget(make_labeled_field_with_help(
                    "Threshold",
                    threshold_edit,
                    "Controls how sensitive the artifact detector is.\n\n"
                    "The algorithm measures scan-to-scan jumps across all voltage columns, then "
                    "flags any jump larger than: median + threshold × MAD.\n\n"
                    "Lower value → more sensitive (catches smaller artifacts).\n"
                    "Higher value → less sensitive (only catches very large artifacts).\n\n"
                    "Start with 8. If the artifact is not being removed, try lowering to 4–6. "
                    "If real signal is being erased, raise to 10–15."
                ))

                pad_layout = QHBoxLayout()
                pad_label = QLabel("Pad (scans):")
                pad_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                pad_edit = QLineEdit("2")
                pad_layout.addWidget(pad_label)
                pad_layout.addWidget(make_labeled_field_with_help(
                    "Pad",
                    pad_edit,
                    "Extra scans added to both edges of the detected artifact window.\n\n"
                    "The artifact onset and offset often include a slow ramp that the jump "
                    "detector may not flag. Adding padding ensures these ramp scans are also "
                    "replaced by the linear interpolation.\n\n"
                    "Default of 2 works for most stimulation artifacts. "
                    "Increase to 3–5 if the artifact edges still look noisy after removal."
                ))

                max_scans_layout = QHBoxLayout()
                max_scans_label = QLabel("Max Artifact Scans (0 = auto):")
                max_scans_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                max_scans_edit = QLineEdit("0")
                max_scans_layout.addWidget(max_scans_label)
                max_scans_layout.addWidget(make_labeled_field_with_help(
                    "Max Artifact Scans",
                    max_scans_edit,
                    "Maximum number of scans that can be classified as a single artifact region.\n\n"
                    "Prevents the algorithm from accidentally removing large sections of real signal "
                    "if a strong response happens to produce a big jump.\n\n"
                    "0 = auto, which sets the limit to 2 × acquisition frequency (≈ 2 seconds).\n\n"
                    "Increase this if your stimulation lasts longer than 2 seconds. "
                    "Decrease it (e.g. to 5–10) if you have very brief stimulations and want "
                    "to be conservative about what gets interpolated."
                ))

                if "Artifact Removal" in saved_params:
                    sp = saved_params["Artifact Removal"]
                    threshold_edit.setText(sp.get("threshold", "8"))
                    pad_edit.setText(sp.get("pad", "2"))
                    max_scans_edit.setText(sp.get("max_artifact_scans", "0"))

                ar_layout.addLayout(threshold_layout)
                ar_layout.addLayout(pad_layout)
                ar_layout.addLayout(max_scans_layout)
                ar_container = QWidget()
                ar_container.setLayout(ar_layout)
                ar_container.setContentsMargins(24, 0, 0, 0)
                ar_container.hide()
                param_widget = ar_container
                self.param_widgets[name] = {
                    "threshold": threshold_edit,
                    "pad": pad_edit,
                    "max_artifact_scans": max_scans_edit,
                }
            elif name == "Multiple Peak Detection":
                # Parameters for multiple peak detection
                mpd_layout = QVBoxLayout()

                # Max peaks
                max_peaks_layout = QHBoxLayout()
                max_peaks_label = QLabel("Max Peaks:")
                max_peaks_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                max_peaks_edit = QLineEdit("10")
                max_peaks_layout.addWidget(max_peaks_label)
                max_peaks_layout.addWidget(max_peaks_edit)

                # Min prominence
                prominence_layout = QHBoxLayout()
                prominence_label = QLabel("Min Prominence:")
                prominence_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                prominence_edit = QLineEdit("0.5")
                prominence_layout.addWidget(prominence_label)
                prominence_layout.addWidget(prominence_edit)

                # Rise window
                min_rise_layout = QHBoxLayout()
                min_rise_label = QLabel("Min Rise Window (sec):")
                min_rise_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                min_rise_edit = QLineEdit("0.2")
                min_rise_layout.addWidget(min_rise_label)
                min_rise_layout.addWidget(min_rise_edit)

                max_rise_layout = QHBoxLayout()
                max_rise_label = QLabel("Min Rise Window (sec):")
                max_rise_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                max_rise_edit = QLineEdit("3.0")
                max_rise_layout.addWidget(max_rise_label)
                max_rise_layout.addWidget(max_rise_edit)

                # Decay window
                min_decay_layout = QHBoxLayout()
                min_decay_label = QLabel("Min Decay Window (sec):")
                min_decay_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                min_decay_edit = QLineEdit("5.0")
                min_decay_layout.addWidget(min_decay_label)
                min_decay_layout.addWidget(min_decay_edit)

                max_decay_layout = QHBoxLayout()
                max_decay_label = QLabel("Max Decay Window (sec):")
                max_decay_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                max_decay_edit = QLineEdit("20.0")
                max_decay_layout.addWidget(max_decay_label)
                max_decay_layout.addWidget(max_decay_edit)

                cv_peak_layout = QHBoxLayout()
                cv_peak_label = QLabel("CV Peak:")
                cv_peak_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                cv_peak_edit = QLineEdit("0.6")
                cv_peak_layout.addWidget(cv_peak_label)
                cv_peak_layout.addWidget(cv_peak_edit)

                peak_height_threshold_layout = QHBoxLayout()
                peak_height_threshold_label = QLabel("Peak Height Threshold:")
                peak_height_threshold_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                peak_height_threshold_edit = QLineEdit("0.2")
                peak_height_threshold_layout.addWidget(peak_height_threshold_label)
                peak_height_threshold_layout.addWidget(peak_height_threshold_edit)


                if "Multiple Peak Detection" in saved_params:
                    params = saved_params["Multiple Peak Detection"]
                    max_peaks_edit.setText(str(params.get("max_peaks", "10")))
                    prominence_edit.setText(str(params.get("min_prominence", "0.5")))
                    min_rise_edit.setText(str(params.get("min_rise_window_sec", "0.2")))
                    max_rise_edit.setText(str(params.get("max_rise_window_sec", "3.0")))
                    min_decay_edit.setText(str(params.get("min_decay_window_sec", "5.0")))
                    max_decay_edit.setText(str(params.get("max_decay_window_sec", "20.0")))
                    cv_peak_edit.setText(str(params.get("cv_peak", "0.6")))
                    peak_height_threshold_edit.setText(str(params.get("peak_height_threshold", "0.2")))

                mpd_layout.addLayout(max_peaks_layout)
                mpd_layout.addLayout(prominence_layout)
                mpd_layout.addLayout(min_rise_layout)
                mpd_layout.addLayout(max_rise_layout)
                mpd_layout.addLayout(min_decay_layout)
                mpd_layout.addLayout(max_decay_layout)
                mpd_layout.addLayout(cv_peak_layout)
                mpd_layout.addLayout(peak_height_threshold_layout)

                mpd_container = QWidget()
                mpd_container.setLayout(mpd_layout)
                mpd_container.setContentsMargins(24, 0, 0, 0)  # Indent
                mpd_container.hide()
                param_widget = mpd_container
                self.param_widgets[name] = {
                    "max_peaks": max_peaks_edit,
                    "min_prominence": prominence_edit,
                    "min_rise_window_sec": min_rise_edit,
                    "max_rise_window_sec": max_rise_edit,
                    "min_decay_window_sec": min_decay_edit,
                    "max_decay_window_sec": max_decay_edit,
                    "peak_height_threshold": peak_height_threshold_edit,
                    "cv_peak": cv_peak_edit,
                }

            # For "Invert Data", no parameters, just a checkbox
            # (No need to disable, just leave unchecked by default)
            elif name == "Butterworth Filter":
                bw_layout = QHBoxLayout()
                bw_label_p = QLabel("Order (p):")
                bw_label_p.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                bw_p = QLineEdit("4")
                bw_label_cx = QLabel("cx:")
                bw_label_cx.setStyleSheet("font-size: 11px; color: #555;")
                bw_cx = QLineEdit("2.5")
                bw_label_cy = QLabel("cy:")
                bw_label_cy.setStyleSheet("font-size: 11px; color: #555;")
                bw_cy = QLineEdit("37500.0")
                if "Butterworth Filter" in saved_params:
                    p, cx, cy = saved_params["Butterworth Filter"]
                    bw_p.setText(p)
                    bw_cx.setText(cx)
                    bw_cy.setText(cy)
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

            # Add parameter widget to filter layout if it exists
            if param_widget:
                filter_layout.addWidget(param_widget)

                # Show/hide parameter widget based on checkbox
                def toggle_widget(checked, widget=param_widget):
                    widget.setVisible(checked)
                cb.stateChanged.connect(toggle_widget)
                # Set initial visibility
                param_widget.setVisible(cb.isChecked())

            # Add the filter layout to the main dialog layout
            filter_container = QWidget()
            filter_container.setLayout(filter_layout)
            layout.addWidget(filter_container)

        single_peak_layout = QHBoxLayout()
        single_peak_layout.setContentsMargins(0, 0, 0, 0)
        single_peak_layout.setSpacing(0)

        self.find_amplitudes_btn = QPushButton("Single Peak Detection: ON")
        self.find_amplitudes_btn.setCheckable(True)
        self.find_amplitudes_btn.setChecked(True)   # Always starts ON
        self.find_amplitudes_btn.setEnabled(False)  # Users can't click directly

        # Apply your green general style
        apply_custom_styles(self.find_amplitudes_btn)

        # Add button flush-left in a horizontal layout
        single_peak_layout.addWidget(self.find_amplitudes_btn)
        single_peak_layout.addStretch()  # Pushes everything else to the left
        layout.addLayout(single_peak_layout)

        # Wire Multiple Peak Detection -> Single Peak Detection button state
        mpd_cb = self.checkboxes.get("Multiple Peak Detection")

        if mpd_cb is not None:
            def sync_single_peak_detection(checked: bool):
                # When MPD is on, Single Peak Detection shows OFF; otherwise ON
                self.find_amplitudes_btn.setChecked(not checked)
                self.find_amplitudes_btn.setText(
                    "Single Peak Detection: OFF" if checked else "Single Peak Detection: ON"
                )
            sync_single_peak_detection(mpd_cb.isChecked())  # Initial state
            mpd_cb.toggled.connect(sync_single_peak_detection)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

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
                p, cx, cy = 4, 0.75, 37500.0
            return ButterworthFilter(p=p, cx=cx, cy=cy)
        elif name == "Baseline Correction":
            return BaselineCorrection()
        elif name == "Normalize":
            return Normalize(peak_position)
        elif name == "Invert Data":
            return InvertData()
        elif name == "Single Peak Detection":
            return FindAmplitude(peak_position)
        elif name == "Multiple Peak Detection":
            params = self.param_widgets[name]
            try:
                max_peaks = int(params["max_peaks"].text())
                min_prominence = float(params["min_prominence"].text())
                min_rise_window_sec = float(params["min_rise_window_sec"].text())
                min_rise_window_sec = float(params["min_rise_window_sec"].text())
                min_decay_window_sec = float(params["min_decay_window_sec"].text())
                max_decay_window_sec = float(params["max_decay_window_sec"].text())
                cv_peak = float(params["cv_peak"].text())
                peak_height_threshold = float(params["peak_height_threshold"].text())

            except ValueError:
                max_peaks = 10
                min_prominence = 0.5
                rise_window_sec = 3.0
                decay_window_sec = 10.0
                cv_peak = 0.6
            return FindAmplitudeMultiple(
                peak_position=peak_position,
                max_peaks=max_peaks,
                min_prominence=min_prominence,
                min_rise_window_sec=min_rise_window_sec,
                min_decay_window_sec=min_decay_window_sec,
                max_decay_window_sec=max_decay_window_sec,
                cv_peak=cv_peak,
                peak_height_threshold=peak_height_threshold,
            )
        elif name == "Exponential Fitting":
            return ExponentialFitting()
        elif name == "Artifact Removal":
            params = self.param_widgets.get(name, {})
            try:
                threshold = float(params["threshold"].text())
                pad = int(params["pad"].text())
                raw_max = int(params["max_artifact_scans"].text())
                max_artifact_scans = None if raw_max == 0 else raw_max
            except (ValueError, KeyError):
                threshold, pad, max_artifact_scans = 8, 2, None
            return StimArtifactRemoval(threshold=threshold, pad=pad, max_artifact_scans=max_artifact_scans)
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
        """
        Saves the selected processor configuration and parameters to QSettings,
        then closes the dialog.
        """
        selected = self.get_selected_processors()
        self.qsettings.setValue("processing_pipeline", json.dumps(selected))

        # now save params
        out = {}
        for name, widget in self.param_widgets.items():
            if name == "Artifact Removal":
                out[name] = {
                    "threshold": widget["threshold"].text(),
                    "pad": widget["pad"].text(),
                    "max_artifact_scans": widget["max_artifact_scans"].text(),
                }
            elif name == "Multiple Peak Detection":
                # Special handling for multiple peak detection parameters
                out[name] = {
                    "max_peaks": widget["max_peaks"].text(),
                    "min_prominence": widget["min_prominence"].text(),
                    "min_rise_window_sec": widget["min_rise_window_sec"].text(),
                    "max_rise_window_sec": widget["max_rise_window_sec"].text(),
                    "min_decay_window_sec": widget["min_decay_window_sec"].text(),
                    "max_decay_window_sec": widget["max_decay_window_sec"].text(),
                    "cv_peak": widget["cv_peak"].text(),
                    "peak_height_threshold": widget["peak_height_threshold"].text(),
                }
            elif isinstance(widget, tuple):
                # multiple-lineEdits
                out[name] = [w.text() for w in widget]
            else:
                out[name] = widget.text()

        self.qsettings.setValue("processing_params", json.dumps(out))

        super().accept()