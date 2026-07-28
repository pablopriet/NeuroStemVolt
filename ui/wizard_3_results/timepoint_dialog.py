from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDialogButtonBox,
    QListWidget, QListWidgetItem, QPushButton
)
from PyQt5.QtCore import Qt

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


class TimepointMultiSelectionDialog(QDialog):
    """
    Dialog to choose WHICH timepoints get an exponential-fit plot exported.

    The exporter used to hard-code timepoint 0, so "Save All Plots" always wrote
    a single fit for the first file. This lets the user tick any subset; every
    timepoint is ticked by default so the common case is one plot per timepoint.

    Args:
        file_names (list of str): one label per timepoint, in file order.
        parent (QWidget, optional): parent widget.
    """
    def __init__(self, file_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Timepoints to Export")
        self.resize(420, 360)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Tick the timepoints to export an exponential-fit plot for:"))

        self.list = QListWidget(self)
        for name in file_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)   # default: export them all
            self.list.addItem(item)
        layout.addWidget(self.list)

        select_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        btn_none = QPushButton("Select None")
        btn_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        select_row.addWidget(btn_all)
        select_row.addWidget(btn_none)
        select_row.addStretch()
        layout.addLayout(select_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, state):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(state)

    def get_selected_indices(self):
        """Indices of the ticked timepoints, in file order."""
        return [i for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]
