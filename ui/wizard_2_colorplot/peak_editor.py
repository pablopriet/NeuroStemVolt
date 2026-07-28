# ui/peak_editor.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QSpinBox,
                             QPushButton, QDialogButtonBox, QCheckBox, QMessageBox)
from PyQt5.QtCore import pyqtSignal

class PeakEditorDialog(QDialog):
    peaks_changed = pyqtSignal(list, int)  # emits (peaks_list, active_index)
    """Edit/add/remove peaks and choose the active one; optional plot-click picking."""
    def __init__(self, peaks, active_idx=0, max_index=10, canvas=None, canvases=None, file_type="Multi-Peak", acq_freq=10.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Peaks")
        # Support one or many Matplotlib canvases (e.g., colour plot + IT plot)
        if canvases is not None and isinstance(canvases, (list, tuple)):
            self._canvases = [c for c in canvases if c is not None]
        elif canvas is not None:
            self._canvases = [canvas]
        else:
            self._canvases = []
        self._mpl_cids = []  # one cid per connected canvas
        self._max_index = int(max_index)
        self._peaks = [int(p) for p in peaks]
        self._active = max(0, min(int(active_idx), len(self._peaks)-1)) if self._peaks else 0
        self._file_type = file_type
        self._acq_freq = float(acq_freq) if acq_freq else 10.0

        v = QVBoxLayout(self)
        v.addWidget(QLabel("Peaks (indices):"))
        self.listw = QListWidget(); v.addWidget(self.listw)

        row = QHBoxLayout(); row.addWidget(QLabel("Index:"))
        self.spin = QSpinBox(); self.spin.setRange(0, self._max_index); row.addWidget(self.spin)
        self.btn_set = QPushButton("Set for Selected"); row.addWidget(self.btn_set)
        v.addLayout(row)

        row2 = QHBoxLayout()
        self.btn_add = QPushButton("Add"); self.btn_remove = QPushButton("Remove"); self.btn_set_active = QPushButton("Set Active")
        for b in (self.btn_add, self.btn_remove, self.btn_set_active):
            row2.addWidget(b)
        v.addLayout(row2)

        self.chk_click = QCheckBox("Pick by clicking on plot(s)"); v.addWidget(self.chk_click)
        self.chk_click.setEnabled(len(self._canvases) > 0)

        self.btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); v.addWidget(self.btns)

        # signals
        self.listw.currentRowChanged.connect(self._on_select)
        self.btn_set.clicked.connect(self._on_set_index)
        self.btn_add.clicked.connect(self._on_add)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_set_active.clicked.connect(self._on_set_active)
        self.btns.accepted.connect(self.accept); self.btns.rejected.connect(self.reject)
        self.chk_click.toggled.connect(self._toggle_canvas_pick)

        self._refresh()
        if self._peaks: self.listw.setCurrentRow(self._active)

    def _emit_change(self):
        # Emit a copy so listeners don't mutate our internal list inadvertently
        self.peaks_changed.emit(list(self._peaks), int(self._active))

    def _refresh(self):
        self.listw.clear()
        for i, p in enumerate(self._peaks):
            lbl = f"{p}" + ("   (active)" if i == self._active else "")
            self.listw.addItem(lbl)
        # Enforce UI constraint for stimulation files: only one peak allowed
        if self._file_type == "Single Peak":
            self.btn_add.setEnabled(len(self._peaks) == 0)
        else:
            self.btn_add.setEnabled(True)

    def _on_select(self, row):
        if 0 <= row < len(self._peaks):
            self.spin.setValue(self._peaks[row])

    def _on_set_index(self):
        row = self.listw.currentRow()
        if 0 <= row < len(self._peaks):
            self._peaks[row] = max(0, min(int(self.spin.value()), self._max_index))
            self._refresh(); self.listw.setCurrentRow(row)
            self._emit_change()

    def _on_add(self):
        v = max(0, min(int(self.spin.value()), self._max_index))
        if self._file_type == "Single Peak":
            if len(self._peaks) >= 1:
                # Replace the current (active or first) peak instead of adding a second
                target = self.listw.currentRow()
                if not (0 <= target < len(self._peaks)):
                    target = self._active if 0 <= self._active < len(self._peaks) else 0
                self._peaks[target] = v
                self._active = target
                self._refresh(); self.listw.setCurrentRow(target)
                self._emit_change()
                return
        self._peaks.append(v)
        if len(self._peaks) == 1: self._active = 0
        self._refresh(); self.listw.setCurrentRow(len(self._peaks)-1)
        self._emit_change()

    def _on_remove(self):
        row = self.listw.currentRow()
        if 0 <= row < len(self._peaks):
            del self._peaks[row]
            self._active = 0 if not self._peaks else max(0, min(self._active, len(self._peaks)-1))
            self._refresh(); self.listw.setCurrentRow(min(row, len(self._peaks)-1))
            self._emit_change()

    def _on_set_active(self):
        row = self.listw.currentRow()
        if 0 <= row < len(self._peaks):
            self._active = row; self._refresh(); self.listw.setCurrentRow(row)
            self._emit_change()

    def _toggle_canvas_pick(self, enabled):
        if not self._canvases:
            return
        # Disconnect any existing first
        for i, (canvas, cid) in enumerate(list(zip(self._canvases, self._mpl_cids))):
            if cid is not None:
                try:
                    canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._mpl_cids = []

        if enabled:
            for canvas in self._canvases:
                try:
                    cid = canvas.mpl_connect("button_press_event", self._on_click)
                except Exception:
                    cid = None
                self._mpl_cids.append(cid)

    def _on_click(self, ev):
        if ev.inaxes is None or ev.xdata is None: return
        # Both colorplot and IT plot now use seconds on x-axis
        # Convert seconds to sample index
        x_sec = max(0.0, float(ev.xdata))
        idx = int(round(x_sec * self._acq_freq))
        idx = max(0, min(idx, self._max_index))
        self.spin.setValue(idx); self._on_set_index()
        self._emit_change()

    def result(self):
        seen, uniq = set(), []
        for p in self._peaks:
            if p not in seen:
                uniq.append(p); seen.add(p)
        # Clamp to a single peak in Stimulation mode
        if self._file_type == "Single Peak" and len(uniq) > 1:
            keep_idx = self._active if 0 <= self._active < len(uniq) else 0
            uniq = [uniq[keep_idx]]
        active = 0 if not uniq else max(0, min(self._active, len(uniq)-1))
        return uniq, active

    def accept(self):
        peaks, _ = self.result()
        if self._file_type == "Single Peak":
            if len(peaks) > 1:
                QMessageBox.warning(self, "Too many peaks", "Only one peak is allowed for Stimulation files.")
                return
        if any(p < 0 or p > self._max_index for p in peaks):
            QMessageBox.warning(self, "Invalid peak", "One or more peaks are out of range.")
            return
        super().accept()