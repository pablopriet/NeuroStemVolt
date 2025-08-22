from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QLineEdit, QHBoxLayout,
    QPushButton, QDialogButtonBox, QFileDialog
)
from PyQt5.QtCore import QSettings
import json

from ui.utils.ui_helpers import make_labeled_field_with_help

class ExperimentSettingsDialog(QDialog):
    """
    Dialog window for configuring experiment-level parameters before analysis.

    This includes metadata such as acquisition frequency, peak position, and stimulation settings.
    Parameters are persisted using QSettings for future sessions.

    Args:
        parent (QWidget, optional): Parent widget.
        defaults (dict, optional): Optional default values to override QSettings.
    """
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Experiment Settings")

        self.qsettings = QSettings("HashemiLab", "NeuroStemVolt")

        defaults = {
            "file_length":           self.qsettings.value("file_length",           100,    type=int),
            "acquisition_frequency": self.qsettings.value("acquisition_frequency", 10,     type=int),
            "peak_position":         self.qsettings.value("peak_position",         257,    type=int),
            "treatment":             self.qsettings.value("treatment",             "None", type=str),
            #"waveform":              self.qsettings.value("waveform",              "5HT",  type=str),
            "time_between_files":    self.qsettings.value("time_between_files",    10,     type=int),
            "files_before_treatment":self.qsettings.value("files_before_treatment",3,      type=int),
            "file_type":             self.qsettings.value("file_type",             "None", type=str),
            # stim_params might be stored as JSON
            "stim_params":           json.loads(self.qsettings.value("stim_params", "{}")),
            "output_folder": self.qsettings.value("output_folder", "", type=str),
        }

        vbox = QVBoxLayout()
        
        # Form layout for labeled fields
        form = QFormLayout()
        vbox.addLayout(form)

        #self.cb_waveform    = QComboBox();  self.cb_waveform.addItems(["5HT","Else"])
        #self.cb_waveform.setCurrentText(defaults["waveform"]);                     form.addRow("Waveform:", self.cb_waveform)

        self.cb_file_type = QComboBox(); self.cb_file_type.addItems(["None","Spontaneous","Stimulation"])
        self.cb_file_type.setCurrentText(defaults["file_type"]);                   form.addRow("File Type:", self.cb_file_type)

        self.le_acq_freq = QLineEdit(str(defaults["acquisition_frequency"]))  
        form.addRow("Acquisition Frequency (Hz):", make_labeled_field_with_help(
            "Acquisition Frequency (Hz)", self.le_acq_freq,
            "Sampling rate of the acquisition system, in Hertz (Hz)."
        ))

        self.le_file_length = QLineEdit(str(defaults["file_length"]))
        form.addRow("File Length (seconds):", make_labeled_field_with_help(
            "File Length (seconds)", self.le_file_length,
            "Total duration (in seconds) of each recorded file."
        ))

        self.le_peak_pos = QLineEdit(str(defaults["peak_position"])) 
        form.addRow("Peak Position:", make_labeled_field_with_help(
            "Peak Position", self.le_peak_pos,
            "Expected position of the signal peak on the voltage axis (e.g., 257 for 5HT). "
            "You may enter an approximate value and adjust it later after identifying the actual peak."
        ))

        self.le_treatment = QLineEdit(defaults["treatment"])
        form.addRow("Treatment:", make_labeled_field_with_help(
            "Treatment", self.le_treatment,
            "Name of the treatment applied (e.g., Sertraline)."
        ))

        self.le_time_btw = QLineEdit(str(defaults["time_between_files"]))
        form.addRow("Time Between Files (minutes):", make_labeled_field_with_help(
            "Time Between Files (minutes)", self.le_time_btw,
            "Interval (in minutes) between each stimulation or recording session (e.g., 10)."
        ))

        self.le_files_before = QLineEdit(str(defaults["files_before_treatment"])) 
        form.addRow("Files Before Treatment:", make_labeled_field_with_help(
            "Files Before Treatment", self.le_files_before,
            "Number of recording files acquired before applying the treatment "
            "(e.g., 3 untreated files, followed by treated ones)."
        ))

        # store loaded stim_params so get_settings() can return it if user doesn’t change it
        self.stim_params = defaults["stim_params"]

        h_output = QHBoxLayout()
        self.le_output_folder = QLineEdit(defaults["output_folder"])
        btn_browse_output = QPushButton("Browse...")
        btn_browse_output.clicked.connect(self.browse_output_folder)
        h_output.addWidget(self.le_output_folder)
        h_output.addWidget(btn_browse_output)
        form.addRow("Output Folder:", h_output)

        self.setLayout(vbox)

        # Add dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        vbox.addWidget(buttons)
        buttons.accepted.connect(self.handle_accept)
        buttons.rejected.connect(self.reject)

    def browse_output_folder(self):
        """
        Opens a QFileDialog to select an output folder and populates the text field.

        Returns:
            None
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.le_output_folder.setText(folder)

    def handle_accept(self):
        """
        Handler for the OK button. Validates input, optionally launches
        the `StimParamsDialog`, and stores all parameters in QSettings.

        Returns:
            None
        """
        # if they choose stimulation, pop the sub-dialog
        if self.cb_file_type.currentText() == "Stimulation":
            dlg = StimParamsDialog(self, defaults=self.stim_params)
            if dlg.exec_() == QDialog.Accepted:
                self.stim_params = dlg.get_params()
            else:
                return  # abort if they cancelled stim-params

        # now persist *all* fields
        self.qsettings.setValue("file_type",             self.cb_file_type.currentText())
        self.qsettings.setValue("acquisition_frequency", int(self.le_acq_freq.text()))
        self.qsettings.setValue("file_length",           int(self.le_file_length.text()))
        self.qsettings.setValue("peak_position",         int(self.le_peak_pos.text()))
        self.qsettings.setValue("treatment",             self.le_treatment.text())
        #self.qsettings.setValue("waveform",              self.cb_waveform.currentText())
        self.qsettings.setValue("time_between_files",    int(self.le_time_btw.text()))
        self.qsettings.setValue("files_before_treatment",int(self.le_files_before.text()))
        self.qsettings.setValue("output_folder", self.le_output_folder.text())
        # stim_params → JSON string
        print("Saving stim_params:", self.stim_params)
        self.qsettings.setValue("stim_params", json.dumps(self.stim_params))
        print("After setValue, reload raw:", self.qsettings.value("stim_params"))

        # close dialog
        self.accept()

    def get_settings(self):
        """
        Extract and return the configured settings from the dialog.

        Returns:
            dict: Dictionary with experiment settings, including:
                - file_length (int)
                - acquisition_frequency (int)
                - peak_position (int)
                - treatment (str)
                - time_between_files (float)
                - files_before_treatment (int)
                - file_type (str)
                - stim_params (dict)
                - output_folder (str)
        """
        return {
            "file_length":            int(self.le_file_length.text()),
            "acquisition_frequency":  int(self.le_acq_freq.text()),
            "peak_position":          int(self.le_peak_pos.text()),
            "treatment":              self.le_treatment.text(),
            #"waveform":               self.cb_waveform.currentText(),
            "time_between_files":     float(self.le_time_btw.text()),
            "files_before_treatment": int(self.le_files_before.text()),
            "file_type":              self.cb_file_type.currentText(),
            "stim_params":            self.stim_params,    # initialized in __init__
            "output_folder": self.le_output_folder.text(),
        }

class StimParamsDialog(QDialog):
    def __init__(self, parent=None, defaults=None):
        """
        Dialog window for entering electrical stimulation parameters.

        Parameters include pulse start time, frequency, amplitude, and count.
        Automatically computes total stimulation duration.

        Args:
            parent (QWidget, optional): Parent widget.
            defaults (dict, optional): Dictionary with default stimulation values.
                Expected keys: 'start', 'frequency', 'amplitude', 'pulses'
        """
        super().__init__(parent)
        self.setWindowTitle("Stimulation Parameters")
        form = QFormLayout(self)
        self.edits = {}
        self.params = ["start", "frequency", "amplitude", "pulses"]
        defaults = defaults or {"start": 5.0, "frequency": 20, "amplitude": 0.5, "pulses": 50}
        help_texts = {
            "start": "Start time of stimulation in minutes.",
            "pulses": "Total number of stimulation pulses.",
            "frequency": "Frequency of stimulation pulses in Hz.",
            "amplitude": "Amplitude of stimulation current in uA.",
        }

        for p in self.params:
            edit = QLineEdit(str(defaults[p]))
            help_widget = make_labeled_field_with_help(p.capitalize(), edit, help_texts[p])
            form.addRow(f"{p.capitalize()}:", help_widget)
            self.edits[p] = edit

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def get_params(self):
        """
        Retrieve stimulation parameters entered by the user.

        Returns:
            dict: Dictionary containing:
                - start (float): Stimulation start time in minutes.
                - frequency (float): Pulse frequency in Hz.
                - amplitude (float): Stimulation amplitude in μA.
                - pulses (float): Number of pulses.
                - duration (float): Calculated stimulation duration in seconds.
        """
        params = {}

        # Get user inputs
        for p in self.params:
            try:
                params[p] = float(self.edits[p].text())
            except ValueError:
                params[p] = 0.0

        # Calculate duration
        try:
            pulses = params["pulses"]
            frequency = params["frequency"]
            params["duration"] = pulses / frequency if frequency != 0 else 0.0
            print(pulses)
            print(frequency)
            print(params["duration"])
        except KeyError:
            params["duration"] = 0.0

        return params
