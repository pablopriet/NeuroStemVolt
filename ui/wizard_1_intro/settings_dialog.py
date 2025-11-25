from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QLineEdit, QHBoxLayout,
    QPushButton, QDialogButtonBox, QFileDialog, QCheckBox, QLabel, QWidget, QMessageBox,
    QListWidget, QListWidgetItem, QSpinBox
)
from PyQt5.QtCore import QSettings
import json

from ui.utils.ui_helpers import make_labeled_field_with_help

class ExperimentSettingsDialog(QDialog):
    """
    Dialog window for configuring experiment-level parameters before analysis.

    This includes metadata such as acquisition frequency, peak position, and stimulation settings.
    When File Type == "Flow Cell", an InjectionParamsDialog is shown on accept to collect
    injection start and length + calibration concentrations and repetitions.
    Certain fields (treatment, time_between_files, files_before_treatment)
    are disabled for Flow Cell to avoid confusion.
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
            "waveform":              self.qsettings.value("waveform",              "5HT",  type=str),
            "time_between_files":    self.qsettings.value("time_between_files",    10,     type=int),
            "files_before_treatment":self.qsettings.value("files_before_treatment",3,      type=int),
            "file_type":             self.qsettings.value("file_type",             "None", type=str),
            # stim_params might be stored as JSON
            "stim_params":           json.loads(self.qsettings.value("stim_params", "{}")),
            "output_folder": self.qsettings.value("output_folder", "", type=str),
            "calibration_enabled": self.qsettings.value("calibration_enabled", False, type=bool),
            "calibration_slope": self.qsettings.value("calibration_slope", 1.0, type=float),
            "calibration_intercept": self.qsettings.value("calibration_intercept", 0.0, type=float),
            # Flow Cell injection defaults
            "injection_start": self.qsettings.value("injection_start", 0.0, type=float),
            "injection_length": self.qsettings.value("injection_length", 0.0, type=float),
            "calibration_concentrations": json.loads(self.qsettings.value("calibration_concentrations", "[]")),
            "repetitions_per_cal": self.qsettings.value("repetitions_per_cal", 1, type=int),
        }

        vbox = QVBoxLayout()
        
        # Form layout for labeled fields
        form = QFormLayout()
        vbox.addLayout(form)

        self.cb_waveform    = QComboBox();  self.cb_waveform.addItems(["5HT","HA"])
        self.cb_waveform.setCurrentText(defaults["waveform"]);                     form.addRow("Waveform:", self.cb_waveform)

        self.cb_file_type = QComboBox(); self.cb_file_type.addItems(["Stimulation", "Spontaneous", "Flow Cell"])
        self.cb_file_type.setCurrentText(defaults["file_type"]);                   form.addRow("File Type:", self.cb_file_type)
        # toggle dependent fields when file type changes
        self.cb_file_type.currentTextChanged.connect(self._on_file_type_changed)

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

        # Calibration Curve Checkbox and fields
        self.cb_calibration = QCheckBox("Calibration Curve (Current → Concentration)")
        self.cb_calibration.setChecked(defaults["calibration_enabled"])
        form.addRow(self.cb_calibration)

        self.le_slope = QLineEdit(str(defaults["calibration_slope"]))
        self.le_intercept = QLineEdit(str(defaults["calibration_intercept"]))

        self.le_slope.setPlaceholderText("Slope")
        self.le_intercept.setPlaceholderText("Y-intercept")

        # Container for calibration fields
        h_calib = QHBoxLayout()
        h_calib.addWidget(QLabel("Slope:"))
        h_calib.addWidget(self.le_slope)
        h_calib.addWidget(QLabel("Y-intercept:"))
        h_calib.addWidget(self.le_intercept)
        self.calib_widget = QWidget()
        self.calib_widget.setLayout(h_calib)
        form.addRow(self.calib_widget)

        # Show/hide calibration fields based on checkbox
        self.calib_widget.setVisible(self.cb_calibration.isChecked())
        self.cb_calibration.stateChanged.connect(
            lambda checked: self.calib_widget.setVisible(bool(checked))
        )

        # store loaded stim_params so get_settings() can return it if user doesn’t change it
        self.stim_params = defaults["stim_params"]

        # store injection params default (will be overwritten if user sets Flow Cell injection dialog)
        self.injection_params = {
            "start": defaults.get("injection_start", 0.0),
            "length": defaults.get("injection_length", 0.0),
            "concentrations": defaults.get("calibration_concentrations", []),
            "repetitions": defaults.get("repetitions_per_cal", 1)
        }

        h_output = QHBoxLayout()
        self.le_output_folder = QLineEdit(defaults["output_folder"])
        btn_browse_output = QPushButton("Browse...")
        btn_browse_output.clicked.connect(self.browse_output_folder)
        h_output.addWidget(self.le_output_folder)
        h_output.addWidget(btn_browse_output)
        form.addRow("Output Folder:", h_output)

        # apply initial enable/disable depending on file type
        self._on_file_type_changed(self.cb_file_type.currentText())

        self.setLayout(vbox)

        # Add dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        vbox.addWidget(buttons)
        buttons.accepted.connect(self.handle_accept)
        buttons.rejected.connect(self.reject)

    def _on_file_type_changed(self, text):
        """
        Disable fields that are irrelevant for Flow Cell experiments to avoid user confusion.
        """
        is_flow = (text == "Flow Cell")
        # disable treatment / time / files_before when Flow Cell is selected
        self.le_treatment.setEnabled(not is_flow)
        self.le_time_btw.setEnabled(not is_flow)
        self.le_files_before.setEnabled(not is_flow)
        # also hide calibration checkbox when Flow Cell
        self.cb_calibration.setEnabled(not is_flow)

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
        the `StimParamsDialog` or `InjectionParamsDialog`, and stores all parameters in QSettings.

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

        # if they choose Flow Cell, pop injection params dialog
        if self.cb_file_type.currentText() == "Flow Cell":
            dlg = InjectionParamsDialog(self, defaults=self.injection_params)
            if dlg.exec_() == QDialog.Accepted:
                self.injection_params = dlg.get_params()
            else:
                return  # abort if they cancelled injection params

        # now persist *all* fields
        self.qsettings.setValue("file_type",             self.cb_file_type.currentText())
        self.qsettings.setValue("acquisition_frequency", int(self.le_acq_freq.text()))
        self.qsettings.setValue("file_length",           int(self.le_file_length.text()))
        self.qsettings.setValue("peak_position",         int(self.le_peak_pos.text()))
        self.qsettings.setValue("treatment",             self.le_treatment.text())
        self.qsettings.setValue("waveform",              self.cb_waveform.currentText())
        self.qsettings.setValue("time_between_files",    int(self.le_time_btw.text()))
        self.qsettings.setValue("files_before_treatment",int(self.le_files_before.text()))
        self.qsettings.setValue("output_folder", self.le_output_folder.text())
        # stim_params → JSON string
        self.qsettings.setValue("stim_params", json.dumps(self.stim_params))

        # persist injection params for Flow Cell (including concentrations and repetitions)
        self.qsettings.setValue("injection_start", float(self.injection_params.get("start", 0.0)))
        self.qsettings.setValue("injection_length", float(self.injection_params.get("length", 0.0)))
        self.qsettings.setValue("calibration_concentrations", json.dumps(self.injection_params.get("concentrations", [])))
        self.qsettings.setValue("repetitions_per_cal", int(self.injection_params.get("repetitions", 1)))

        self.qsettings.setValue("calibration_enabled", self.cb_calibration.isChecked())
        if self.cb_calibration.isChecked():
            try:
                slope = float(self.le_slope.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Slope must be a number. Defaulting to 1.0.")
                slope = 1.0
            try:
                intercept = float(self.le_intercept.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Y-intercept must be a number. Defaulting to 0.0.")
                intercept = 0.0
            self.qsettings.setValue("calibration_slope", slope)
            self.qsettings.setValue("calibration_intercept", intercept)
        else:
            self.qsettings.setValue("calibration_slope", 1.0)
            self.qsettings.setValue("calibration_intercept", 0.0)

        slope = QSettings("HashemiLab", "NeuroStemVolt").value("calibration_slope", type=float)
        intercept = QSettings("HashemiLab", "NeuroStemVolt").value("calibration_intercept", type=float)

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
                - injection_start, injection_length (floats; present when Flow Cell chosen)
        """
        return {
            "file_length":            int(self.le_file_length.text()),
            "acquisition_frequency":  int(self.le_acq_freq.text()),
            "peak_position":          int(self.le_peak_pos.text()),
            "treatment":              self.le_treatment.text(),
            "waveform":               self.cb_waveform.currentText(),
            "time_between_files":     float(self.le_time_btw.text()) if self.le_time_btw.isEnabled() else 0.0,
            "files_before_treatment": int(self.le_files_before.text()) if self.le_files_before.isEnabled() else 0,
            "file_type":              self.cb_file_type.currentText(),
            "stim_params":            self.stim_params,    # initialized in __init__
            "output_folder": self.le_output_folder.text(),
            "calibration_enabled": self.cb_calibration.isChecked(),
            "calibration_slope": float(self.le_slope.text()) if self.cb_calibration.isChecked() else 1.0,
            "calibration_intercept": float(self.le_intercept.text()) if self.cb_calibration.isChecked() else 0.0,
            "injection_start": float(self.injection_params.get("start", 0.0)) if self.cb_file_type.currentText() == "Flow Cell" else None,
            "injection_length": float(self.injection_params.get("length", 0.0)) if self.cb_file_type.currentText() == "Flow Cell" else None,
            "calibration_concentrations": list(self.injection_params.get("concentrations", [])) if self.cb_file_type.currentText() == "Flow Cell" else None,
            "repetitions_per_cal": int(self.injection_params.get("repetitions", 1)) if self.cb_file_type.currentText() == "Flow Cell" else None,
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
        except KeyError:
            params["duration"] = 0.0

        return params


class InjectionParamsDialog(QDialog):
    """
    Dialog to collect Flow Cell injection parameters:
    - start (seconds or minutes depending on your convention)
    - length
    - a list of calibration concentrations (user-editable)
    - number of repetitions per concentration
    """
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Injection & Calibration Parameters")
        form = QFormLayout(self)
        defaults = defaults or {"start": 0.0, "length": 0.0, "concentrations": [], "repetitions": 1}

        # Start / Length
        self.le_start = QLineEdit(str(defaults.get("start", 0.0)))
        self.le_length = QLineEdit(str(defaults.get("length", 0.0)))
        form.addRow("Injection Start (seconds):", self.le_start)
        form.addRow("Injection Length (seconds):", self.le_length)

        # Concentrations list widget + controls
        self.lst_concs = QListWidget()
        for c in defaults.get("concentrations", []):
            item = QListWidgetItem(str(c))
            self.lst_concs.addItem(item)

        h_add = QHBoxLayout()
        self.le_new_conc = QLineEdit()
        self.le_new_conc.setPlaceholderText("e.g. 500 or 250.0")
        btn_add = QPushButton("Add Concentration")
        btn_remove = QPushButton("Remove Selected")
        h_add.addWidget(self.le_new_conc)
        h_add.addWidget(btn_add)
        h_add.addWidget(btn_remove)

        form.addRow(QLabel("Calibration Concentrations (units):"))
        form.addRow(self.lst_concs)
        form.addRow(h_add)

        btn_add.clicked.connect(self._on_add_conc)
        btn_remove.clicked.connect(self._on_remove_selected)

        # repetitions per concentration
        self.spin_reps = QSpinBox()
        self.spin_reps.setMinimum(1)
        self.spin_reps.setMaximum(100)
        self.spin_reps.setValue(int(defaults.get("repetitions", 1)))
        form.addRow("Repetitions per concentration:", self.spin_reps)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_add_conc(self):
        text = self.le_new_conc.text().strip()
        if not text:
            return
        try:
            # accept floats or ints
            val = float(text) if ('.' in text or 'e' in text.lower()) else int(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid concentration", "Please enter a numeric concentration.")
            return
        item = QListWidgetItem(str(val))
        self.lst_concs.addItem(item)
        self.le_new_conc.clear()

    def _on_remove_selected(self):
        for it in self.lst_concs.selectedItems():
            self.lst_concs.takeItem(self.lst_concs.row(it))

    def _on_ok(self):
        # validate at least one concentration present (optional)
        # accept regardless; caller may check
        self.accept()

    def get_params(self):
        try:
            start = float(self.le_start.text())
        except ValueError:
            start = 0.0
        try:
            length = float(self.le_length.text())
        except ValueError:
            length = 0.0
        # collect concentrations as floats where possible
        concs = []
        for i in range(self.lst_concs.count()):
            txt = self.lst_concs.item(i).text()
            try:
                v = float(txt) if ('.' in txt or 'e' in txt.lower()) else int(txt)
            except ValueError:
                continue
            concs.append(v)
        reps = int(self.spin_reps.value())
        return {"start": start, "length": length, "concentrations": concs, "repetitions": reps}