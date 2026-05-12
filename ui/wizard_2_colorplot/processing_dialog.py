from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QLineEdit, QDialogButtonBox, QWidget, QPushButton
)
from PyQt5.QtCore import QSettings
import json

from ui.utils.styles import apply_custom_styles
from ui.utils.ui_helpers import make_labeled_field_with_help
from core.processing import BackgroundSubtraction, SavitzkyGolayFilter, RollingMean, GaussianSmoothing2D, ButterworthFilter, BaselineCorrection, Normalize, FindAmplitude, ExponentialFitting

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
            ("Find Amplitude", True),
            #("Multiple Peak Detection", False),  # New option
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
            "Butterworth Filter": "Applies a low-pass filter while preserving waveform.",
            "Savitzky-Golay Filter": "Fits a local polynomial of a given 'order' over each segment of the data to smooth noise. The 'window size' sets how many points are used per fit, while 'order' (the 'p' polynomial order) controls how closely the fit can follow rapid changes.",
            "Baseline Correction": "Removes baseline drift from the signal.",
            "Normalize": "Normalizes each trace based on the peak amplitude of the first file within each replicate.",
            #"Multiple Peak Detection": "Detects multiple spontaneous peaks throughout the signal using adaptive validation windows. Useful for analyzing spontaneous activity patterns.",
        }

        for name, default_checked in self.processor_options:
            if name == "Find Amplitude":
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
                rise_layout = QHBoxLayout()
                rise_label = QLabel("Rise Window (sec):")
                rise_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                rise_edit = QLineEdit("3.0")
                rise_layout.addWidget(rise_label)
                rise_layout.addWidget(rise_edit)

                # Decay window
                decay_layout = QHBoxLayout()
                decay_label = QLabel("Decay Window (sec):")
                decay_label.setStyleSheet("font-size: 11px; color: #555; margin-left: 16px;")
                decay_edit = QLineEdit("10.0")
                decay_layout.addWidget(decay_label)
                decay_layout.addWidget(decay_edit)

                if "Multiple Peak Detection" in saved_params:
                    params = saved_params["Multiple Peak Detection"]
                    max_peaks_edit.setText(str(params.get("max_peaks", "10")))
                    prominence_edit.setText(str(params.get("min_prominence", "0.5")))
                    rise_edit.setText(str(params.get("rise_window_sec", "3.0")))
                    decay_edit.setText(str(params.get("decay_window_sec", "10.0")))

                mpd_layout.addLayout(max_peaks_layout)
                mpd_layout.addLayout(prominence_layout)
                mpd_layout.addLayout(rise_layout)
                mpd_layout.addLayout(decay_layout)

                mpd_container = QWidget()
                mpd_container.setLayout(mpd_layout)
                mpd_container.setContentsMargins(24, 0, 0, 0)  # Indent
                mpd_container.hide()
                param_widget = mpd_container
                self.param_widgets[name] = {
                    "max_peaks": max_peaks_edit,
                    "min_prominence": prominence_edit,
                    "rise_window_sec": rise_edit,
                    "decay_window_sec": decay_edit
                }

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
            return ButterworthFilter()
        elif name == "Baseline Correction":
            return BaselineCorrection()
        elif name == "Normalize":
            return Normalize(peak_position)
        elif name == "Find Amplitude":
            # Check file type to determine which amplitude finder to use
            settings = QSettings("HashemiLab", "NeuroStemVolt")
            file_type = settings.value("file_type", "None", type=str)

            if file_type == "Spontaneous":
                return FindAmplitudeMultiple(peak_position)
            else:
                return FindAmplitude(peak_position)
        elif name == "Multiple Peak Detection":
            params = self.param_widgets[name]
            try:
                max_peaks = int(params["max_peaks"].text())
                min_prominence = float(params["min_prominence"].text())
                rise_window_sec = float(params["rise_window_sec"].text())
                decay_window_sec = float(params["decay_window_sec"].text())
            except ValueError:
                max_peaks = 10
                min_prominence = 0.5
                rise_window_sec = 3.0
                decay_window_sec = 10.0
            return FindAmplitudeMultiple(
                peak_position=peak_position,
                max_peaks=max_peaks,
                min_prominence=min_prominence,
                rise_window_sec=rise_window_sec,
                decay_window_sec=decay_window_sec
            )
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
        """
        Saves the selected processor configuration and parameters to QSettings,
        then closes the dialog.
        """
        selected = self.get_selected_processors()
        self.qsettings.setValue("processing_pipeline", json.dumps(selected))

        # now save params
        out = {}
        for name, widget in self.param_widgets.items():
            if name == "Multiple Peak Detection":
                # Special handling for multiple peak detection parameters
                out[name] = {
                    "max_peaks": widget["max_peaks"].text(),
                    "min_prominence": widget["min_prominence"].text(),
                    "rise_window_sec": widget["rise_window_sec"].text(),
                    "decay_window_sec": widget["decay_window_sec"].text()
                }
            elif isinstance(widget, tuple):
                # multiple-lineEdits
                out[name] = [w.text() for w in widget]
            else:
                out[name] = widget.text()

        self.qsettings.setValue("processing_params", json.dumps(out))

        super().accept()