from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QDialogButtonBox

class TimepointSelectionDialog(QDialog):
    """
    Dialog window to allow the user to select a timepoint (file) for exponential fitting.

    This dialog presents a dropdown (QComboBox) populated with the names of available
    FSCV files, allowing the user to choose a specific one for downstream analysis.
    """
    def __init__(self, file_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Timepoint for Exponential Fit")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the file/timepoint to fit:"))
        self.combo = QComboBox(self)
        self.combo.addItems(file_names)
        layout.addWidget(self.combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_index(self):
        return self.combo.currentIndex()
