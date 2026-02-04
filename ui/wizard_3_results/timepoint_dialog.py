from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QDialogButtonBox, QListWidget, QListWidgetItem, QPushButton,
                             QColorDialog, QWidget, QCheckBox, QScrollArea, QFrame, QLineEdit)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QColor
import os
import json

def get_experiment_display_name(experiment, index):
    """
    Get a display name for an experiment based on its folder name.
    
    Args:
        experiment: SpheroidExperiment instance
        index: Index of the experiment in the group
        
    Returns:
        str: Display name (folder name or fallback to index)
    """
    try:
        # Get the first file's path and extract the folder name
        if experiment.files and len(experiment.files) > 0:
            first_file_path = experiment.files[0].get_filepath()
            folder_path = os.path.dirname(first_file_path)
            folder_name = os.path.basename(folder_path)
            if folder_name:
                return folder_name
    except Exception:
        pass
    
    # Fallback to treatment name or index
    treatment = getattr(experiment, 'treatment', None)
    if treatment and treatment.strip():
        return f"Experiment {index+1}: {treatment}"
    return f"Experiment {index+1}"


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


class ExperimentSelectionDialog(QDialog):
    """
    Dialog window to allow the user to select an experiment from the group analysis.

    This dialog presents a dropdown (QComboBox) populated with the names/indices of
    available experiments, allowing the user to choose which experiment to visualize.
    """
    def __init__(self, experiment_names, parent=None, title="Select Experiment"):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select an experiment to visualize:"))
        self.combo = QComboBox(self)
        self.combo.addItems(experiment_names)
        layout.addWidget(self.combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_index(self):
        return self.combo.currentIndex()
    
    def get_selected_name(self):
        return self.combo.currentText()


class ColorButton(QPushButton):
    """A button that displays and allows selection of a color."""
    def __init__(self, color='#1f77b4', parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(30, 25)
        self.update_style()
        self.clicked.connect(self.choose_color)
    
    def update_style(self):
        self.setStyleSheet(f"background-color: {self._color.name()}; border: 1px solid #888;")
    
    def choose_color(self):
        color = QColorDialog.getColor(self._color, self.parent(), "Select Color")
        if color.isValid():
            self._color = color
            self.update_style()
    
    def get_color(self):
        return self._color.name()
    
    def set_color(self, color):
        self._color = QColor(color)
        self.update_style()


class MultiExperimentTimepointDialog(QDialog):
    """
    Dialog for selecting a timepoint and multiple experiments with custom colors.
    
    Allows the user to:
    1. Select a specific timepoint (e.g., -30 min, 0 min, 10 min)
    2. Select multiple experiments to compare
    3. Choose a custom color for each selected experiment
    """
    
    DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def __init__(self, experiments, time_between_files, files_before_treatment, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare ITs at Timepoint")
        self.setMinimumWidth(450)
        self.setMinimumHeight(400)
        
        self.experiments = experiments
        self.time_between_files = time_between_files
        self.files_before_treatment = files_before_treatment
        self.experiment_widgets = []  # List of (checkbox, color_button, exp_index)
        
        layout = QVBoxLayout(self)
        
        # Timepoint selection
        tp_layout = QHBoxLayout()
        tp_layout.addWidget(QLabel("Select Timepoint:"))
        self.timepoint_combo = QComboBox()
        self._populate_timepoints()
        tp_layout.addWidget(self.timepoint_combo)
        layout.addLayout(tp_layout)
        
        # Experiment selection with colors
        layout.addWidget(QLabel("Select experiments and colors:"))
        
        # Scrollable area for experiments
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        exp_container = QWidget()
        exp_layout = QVBoxLayout(exp_container)
        exp_layout.setContentsMargins(0, 0, 0, 0)
        
        for i, exp in enumerate(experiments):
            exp_widget = QWidget()
            exp_h_layout = QHBoxLayout(exp_widget)
            exp_h_layout.setContentsMargins(5, 2, 5, 2)
            
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # Default to selected
            
            color_btn = ColorButton(self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)])
            
            exp_name = get_experiment_display_name(exp, i)
            label = QLabel(exp_name)
            label.setWordWrap(True)
            
            exp_h_layout.addWidget(checkbox)
            exp_h_layout.addWidget(color_btn)
            exp_h_layout.addWidget(label, stretch=1)
            
            exp_layout.addWidget(exp_widget)
            self.experiment_widgets.append((checkbox, color_btn, i))
        
        exp_layout.addStretch()
        scroll.setWidget(exp_container)
        layout.addWidget(scroll, stretch=1)
        
        # Select/Deselect all buttons
        select_layout = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_deselect_all = QPushButton("Deselect All")
        btn_select_all.clicked.connect(self._select_all)
        btn_deselect_all.clicked.connect(self._deselect_all)
        select_layout.addWidget(btn_select_all)
        select_layout.addWidget(btn_deselect_all)
        layout.addLayout(select_layout)
        
        # Load saved settings
        self._load_settings()
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _populate_timepoints(self):
        """Populate timepoint combo with available timepoints in minutes."""
        if not self.experiments:
            return
        
        n_files = self.experiments[0].get_file_count()
        for i in range(n_files):
            # Calculate time relative to treatment
            time_min = (i - self.files_before_treatment) * self.time_between_files
            if time_min >= 0:
                label = f"{time_min:.0f} min"
            else:
                label = f"{time_min:.0f} min"
            self.timepoint_combo.addItem(label, i)  # Store file index as data
    
    def _select_all(self):
        for checkbox, _, _ in self.experiment_widgets:
            checkbox.setChecked(True)
    
    def _deselect_all(self):
        for checkbox, _, _ in self.experiment_widgets:
            checkbox.setChecked(False)
    
    def get_selected_timepoint_index(self):
        """Return the file index for the selected timepoint."""
        return self.timepoint_combo.currentData()
    
    def get_selected_timepoint_label(self):
        """Return the label of the selected timepoint."""
        return self.timepoint_combo.currentText()
    
    def get_selected_experiments(self):
        """Return list of (experiment_index, color) for selected experiments."""
        selected = []
        for checkbox, color_btn, exp_idx in self.experiment_widgets:
            if checkbox.isChecked():
                selected.append((exp_idx, color_btn.get_color()))
        return selected

    def _save_settings(self):
        """Save current dialog settings to QSettings for persistence."""
        settings = QSettings("NeuroStemVolt", "MultiExperimentTimepointDialog")
        
        # Save timepoint index
        settings.setValue("timepoint_index", self.timepoint_combo.currentIndex())
        
        # Save experiment selections and colors
        exp_data = []
        for checkbox, color_btn, exp_idx in self.experiment_widgets:
            exp_data.append({
                'exp_idx': exp_idx,
                'checked': checkbox.isChecked(),
                'color': color_btn.get_color()
            })
        
        settings.setValue("exp_data", json.dumps(exp_data))
    
    def _load_settings(self):
        """Load saved settings from QSettings."""
        settings = QSettings("NeuroStemVolt", "MultiExperimentTimepointDialog")
        
        # Load timepoint index
        tp_index = settings.value("timepoint_index", None)
        if tp_index is not None:
            idx = int(tp_index)
            if 0 <= idx < self.timepoint_combo.count():
                self.timepoint_combo.setCurrentIndex(idx)
        
        # Load experiment data
        exp_json = settings.value("exp_data", None)
        if exp_json:
            try:
                exp_data = json.loads(exp_json)
                
                # Create a mapping from exp_idx to saved data
                saved_map = {item['exp_idx']: item for item in exp_data}
                
                for checkbox, color_btn, exp_idx in self.experiment_widgets:
                    if exp_idx in saved_map:
                        checkbox.setChecked(saved_map[exp_idx].get('checked', True))
                        color_btn.set_color(saved_map[exp_idx].get('color', self.DEFAULT_COLORS[exp_idx % len(self.DEFAULT_COLORS)]))
                        
            except Exception as e:
                print(f"Warning: Could not load saved settings: {e}")
    
    def accept(self):
        """Override accept to save settings before closing."""
        self._save_settings()
        super().accept()


class MultiExperimentAmplitudeDialog(QDialog):
    """
    Dialog for selecting multiple experiments to compare amplitude timelines.
    
    Allows the user to:
    1. Select multiple experiments to compare
    2. Choose a custom color for each selected experiment
    """
    
    DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def __init__(self, experiments, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare Amplitudes Over Time")
        self.setMinimumWidth(450)
        self.setMinimumHeight(350)
        
        self.experiments = experiments
        self.experiment_widgets = []  # List of (checkbox, color_button, exp_index)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        layout.addWidget(QLabel("Select experiments to compare and choose colors:"))
        
        # Scrollable area for experiments
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        exp_container = QWidget()
        exp_layout = QVBoxLayout(exp_container)
        exp_layout.setContentsMargins(0, 0, 0, 0)
        
        for i, exp in enumerate(experiments):
            exp_widget = QWidget()
            exp_h_layout = QHBoxLayout(exp_widget)
            exp_h_layout.setContentsMargins(5, 2, 5, 2)
            
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # Default to selected
            
            color_btn = ColorButton(self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)])
            
            exp_name = get_experiment_display_name(exp, i)
            label = QLabel(exp_name)
            label.setWordWrap(True)
            
            exp_h_layout.addWidget(checkbox)
            exp_h_layout.addWidget(color_btn)
            exp_h_layout.addWidget(label, stretch=1)
            
            exp_layout.addWidget(exp_widget)
            self.experiment_widgets.append((checkbox, color_btn, i))
        
        exp_layout.addStretch()
        scroll.setWidget(exp_container)
        layout.addWidget(scroll, stretch=1)
        
        # Select/Deselect all buttons
        select_layout = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_deselect_all = QPushButton("Deselect All")
        btn_select_all.clicked.connect(self._select_all)
        btn_deselect_all.clicked.connect(self._deselect_all)
        select_layout.addWidget(btn_select_all)
        select_layout.addWidget(btn_deselect_all)
        layout.addLayout(select_layout)
        
        # Load saved settings
        self._load_settings()
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _select_all(self):
        for checkbox, _, _ in self.experiment_widgets:
            checkbox.setChecked(True)
    
    def _deselect_all(self):
        for checkbox, _, _ in self.experiment_widgets:
            checkbox.setChecked(False)
    
    def get_selected_experiments(self):
        """Return list of (experiment_index, color) for selected experiments."""
        selected = []
        for checkbox, color_btn, exp_idx in self.experiment_widgets:
            if checkbox.isChecked():
                selected.append((exp_idx, color_btn.get_color()))
        return selected

    def _save_settings(self):
        """Save current dialog settings to QSettings for persistence."""
        settings = QSettings("NeuroStemVolt", "MultiExperimentAmplitudeDialog")
        
        # Save experiment selections and colors
        exp_data = []
        for checkbox, color_btn, exp_idx in self.experiment_widgets:
            exp_data.append({
                'exp_idx': exp_idx,
                'checked': checkbox.isChecked(),
                'color': color_btn.get_color()
            })
        
        settings.setValue("exp_data", json.dumps(exp_data))
    
    def _load_settings(self):
        """Load saved settings from QSettings."""
        settings = QSettings("NeuroStemVolt", "MultiExperimentAmplitudeDialog")
        
        # Load experiment data
        exp_json = settings.value("exp_data", None)
        if exp_json:
            try:
                exp_data = json.loads(exp_json)
                
                # Create a mapping from exp_idx to saved data
                saved_map = {item['exp_idx']: item for item in exp_data}
                
                for checkbox, color_btn, exp_idx in self.experiment_widgets:
                    if exp_idx in saved_map:
                        checkbox.setChecked(saved_map[exp_idx].get('checked', True))
                        color_btn.set_color(saved_map[exp_idx].get('color', self.DEFAULT_COLORS[exp_idx % len(self.DEFAULT_COLORS)]))
                        
            except Exception as e:
                print(f"Warning: Could not load saved settings: {e}")
    
    def accept(self):
        """Override accept to save settings before closing."""
        self._save_settings()
        super().accept()


class GroupedExperimentDialog(QDialog):
    """
    Dialog for selecting experiments with grouping, timepoint, and custom colors.
    
    Allows the user to:
    1. Select experiments and group them together (same condition)
    2. Select a specific timepoint for comparison
    3. Choose a custom color for each group
    4. Name each group
    
    Used for comparing tau or amplitudes across grouped experiments.
    """
    
    DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def __init__(self, experiments, time_between_files, files_before_treatment, 
                 parent=None, title="Compare Across Groups", metric_name="Tau"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(550)
        self.setMinimumHeight(500)
        
        self.experiments = experiments
        self.time_between_files = time_between_files
        self.files_before_treatment = files_before_treatment
        self.metric_name = metric_name
        self.groups = []  # List of group widgets
        
        layout = QVBoxLayout(self)
        
        # Timepoint selection
        tp_layout = QHBoxLayout()
        tp_layout.addWidget(QLabel("Select Timepoint:"))
        self.timepoint_combo = QComboBox()
        self._populate_timepoints()
        tp_layout.addWidget(self.timepoint_combo)
        layout.addLayout(tp_layout)
        
        # Groups section
        layout.addWidget(QLabel("Create groups and select experiments for each:"))
        
        # Scrollable area for groups
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        self.groups_container = QWidget()
        self.groups_layout = QVBoxLayout(self.groups_container)
        self.groups_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll.setWidget(self.groups_container)
        layout.addWidget(scroll, stretch=1)
        
        # Add/Remove group buttons
        group_btn_layout = QHBoxLayout()
        btn_add_group = QPushButton("+ Add Group")
        btn_add_group.clicked.connect(self._add_group)
        group_btn_layout.addWidget(btn_add_group)
        group_btn_layout.addStretch()
        layout.addLayout(group_btn_layout)
        
        # Add initial two groups (or load from saved settings)
        self._add_group()
        self._add_group()
        
        # Load saved settings (overrides defaults)
        self._load_settings()
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _populate_timepoints(self):
        """Populate timepoint combo with available timepoints in minutes."""
        if not self.experiments:
            return
        
        n_files = self.experiments[0].get_file_count()
        for i in range(n_files):
            time_min = (i - self.files_before_treatment) * self.time_between_files
            label = f"{time_min:.0f} min"
            self.timepoint_combo.addItem(label, i)
    
    def _add_group(self):
        """Add a new group widget."""
        group_idx = len(self.groups)
        
        group_frame = QFrame()
        group_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        group_frame.setStyleSheet("QFrame { background-color: #f5f5f5; border-radius: 5px; }")
        
        group_layout = QVBoxLayout(group_frame)
        group_layout.setContentsMargins(10, 10, 10, 10)
        
        # Group header with name, color, and remove button
        header_layout = QHBoxLayout()
        
        name_edit = QLineEdit(f"Group {group_idx + 1}")
        name_edit.setPlaceholderText("Group name")
        name_edit.setMaximumWidth(150)
        
        color_btn = ColorButton(self.DEFAULT_COLORS[group_idx % len(self.DEFAULT_COLORS)])
        
        btn_remove = QPushButton("×")
        btn_remove.setFixedSize(25, 25)
        btn_remove.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        btn_remove.clicked.connect(lambda checked, f=group_frame: self._remove_group(f))
        
        header_layout.addWidget(QLabel("Name:"))
        header_layout.addWidget(name_edit)
        header_layout.addWidget(QLabel("Color:"))
        header_layout.addWidget(color_btn)
        header_layout.addStretch()
        header_layout.addWidget(btn_remove)
        
        group_layout.addLayout(header_layout)
        
        # Experiment checkboxes
        exp_layout = QVBoxLayout()
        exp_checkboxes = []
        
        for i, exp in enumerate(self.experiments):
            checkbox = QCheckBox(get_experiment_display_name(exp, i))
            exp_layout.addWidget(checkbox)
            exp_checkboxes.append((checkbox, i))
        
        group_layout.addLayout(exp_layout)
        
        self.groups_layout.addWidget(group_frame)
        self.groups.append({
            'frame': group_frame,
            'name_edit': name_edit,
            'color_btn': color_btn,
            'checkboxes': exp_checkboxes
        })
    
    def _remove_group(self, frame):
        """Remove a group widget."""
        for i, group in enumerate(self.groups):
            if group['frame'] == frame:
                self.groups_layout.removeWidget(frame)
                frame.deleteLater()
                self.groups.pop(i)
                break
    
    def get_selected_timepoint_index(self):
        """Return the file index for the selected timepoint."""
        return self.timepoint_combo.currentData()
    
    def get_selected_timepoint_label(self):
        """Return the label of the selected timepoint."""
        return self.timepoint_combo.currentText()
    
    def get_groups(self):
        """
        Return list of groups with their experiments and colors.
        Each group is a dict: {'name': str, 'color': str, 'experiments': [indices]}
        """
        result = []
        for group in self.groups:
            selected_exps = [exp_idx for checkbox, exp_idx in group['checkboxes'] if checkbox.isChecked()]
            if selected_exps:  # Only include groups with at least one experiment
                result.append({
                    'name': group['name_edit'].text(),
                    'color': group['color_btn'].get_color(),
                    'experiments': selected_exps
                })
        return result

    def _save_settings(self):
        """Save current dialog settings to QSettings for persistence."""
        settings = QSettings("NeuroStemVolt", "GroupedExperimentDialog")
        
        # Save timepoint index
        settings.setValue("timepoint_index", self.timepoint_combo.currentIndex())
        
        # Save groups info as JSON
        groups_data = []
        for group in self.groups:
            group_info = {
                'name': group['name_edit'].text(),
                'color': group['color_btn'].get_color(),
                'checked_experiments': [exp_idx for checkbox, exp_idx in group['checkboxes'] if checkbox.isChecked()]
            }
            groups_data.append(group_info)
        
        settings.setValue("groups_data", json.dumps(groups_data))
    
    def _load_settings(self):
        """Load saved settings from QSettings."""
        settings = QSettings("NeuroStemVolt", "GroupedExperimentDialog")
        
        # Load timepoint index
        tp_index = settings.value("timepoint_index", None)
        if tp_index is not None:
            idx = int(tp_index)
            if 0 <= idx < self.timepoint_combo.count():
                self.timepoint_combo.setCurrentIndex(idx)
        
        # Load groups data
        groups_json = settings.value("groups_data", None)
        if groups_json:
            try:
                groups_data = json.loads(groups_json)
                
                # Remove default groups and recreate from saved data
                while self.groups:
                    self._remove_group(self.groups[0]['frame'])
                
                for group_info in groups_data:
                    self._add_group()
                    group = self.groups[-1]
                    
                    # Restore name
                    group['name_edit'].setText(group_info.get('name', f"Group {len(self.groups)}"))
                    
                    # Restore color
                    group['color_btn'].set_color(group_info.get('color', self.DEFAULT_COLORS[len(self.groups) - 1]))
                    
                    # Restore checked experiments
                    checked_exps = set(group_info.get('checked_experiments', []))
                    for checkbox, exp_idx in group['checkboxes']:
                        checkbox.setChecked(exp_idx in checked_exps)
                        
            except Exception as e:
                print(f"Warning: Could not load saved settings: {e}")
    
    def accept(self):
        """Override accept to save settings before closing."""
        self._save_settings()
        super().accept()


class GroupedAmplitudeDialog(QDialog):
    """
    Dialog for selecting experiments with grouping and a single timepoint for amplitude comparison.
    
    Allows the user to:
    1. Select a specific timepoint
    2. Create multiple groups with custom names
    3. Select experiments for each group
    4. Choose a custom color for each group
    
    Used for comparing amplitudes across grouped experiments at a single timepoint.
    """
    
    DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def __init__(self, experiments, time_between_files, files_before_treatment, 
                 parent=None, title="Compare Amplitudes Across Groups"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(550)
        self.setMinimumHeight(500)
        
        self.experiments = experiments
        self.time_between_files = time_between_files
        self.files_before_treatment = files_before_treatment
        self.groups = []
        
        layout = QVBoxLayout(self)
        
        # Timepoint selection (single dropdown)
        tp_layout = QHBoxLayout()
        tp_layout.addWidget(QLabel("Select Timepoint:"))
        self.timepoint_combo = QComboBox()
        self._populate_timepoints()
        tp_layout.addWidget(self.timepoint_combo)
        layout.addLayout(tp_layout)
        
        # Groups section
        layout.addWidget(QLabel("Create groups and select experiments for each:"))
        
        # Scrollable area for groups
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        self.groups_container = QWidget()
        self.groups_layout = QVBoxLayout(self.groups_container)
        self.groups_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll.setWidget(self.groups_container)
        layout.addWidget(scroll, stretch=1)
        
        # Add group button
        group_btn_layout = QHBoxLayout()
        btn_add_group = QPushButton("+ Add Group")
        btn_add_group.clicked.connect(self._add_group)
        group_btn_layout.addWidget(btn_add_group)
        group_btn_layout.addStretch()
        layout.addLayout(group_btn_layout)
        
        # Add initial two groups (or load from saved settings)
        self._add_group()
        self._add_group()
        
        # Load saved settings (overrides defaults)
        self._load_settings()
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _populate_timepoints(self):
        """Populate timepoint combo with available timepoints in minutes."""
        if not self.experiments:
            return
        
        n_files = self.experiments[0].get_file_count()
        for i in range(n_files):
            time_min = (i - self.files_before_treatment) * self.time_between_files
            label = f"{time_min:.0f} min"
            self.timepoint_combo.addItem(label, i)
    
    def _add_group(self):
        """Add a new group widget."""
        group_idx = len(self.groups)
        
        group_frame = QFrame()
        group_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        group_frame.setStyleSheet("QFrame { background-color: #f5f5f5; border-radius: 5px; }")
        
        group_layout = QVBoxLayout(group_frame)
        group_layout.setContentsMargins(10, 10, 10, 10)
        
        # Group header
        header_layout = QHBoxLayout()
        
        name_edit = QLineEdit(f"Group {group_idx + 1}")
        name_edit.setPlaceholderText("Group name")
        name_edit.setMaximumWidth(150)
        
        color_btn = ColorButton(self.DEFAULT_COLORS[group_idx % len(self.DEFAULT_COLORS)])
        
        btn_remove = QPushButton("×")
        btn_remove.setFixedSize(25, 25)
        btn_remove.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        btn_remove.clicked.connect(lambda checked, f=group_frame: self._remove_group(f))
        
        header_layout.addWidget(QLabel("Name:"))
        header_layout.addWidget(name_edit)
        header_layout.addWidget(QLabel("Color:"))
        header_layout.addWidget(color_btn)
        header_layout.addStretch()
        header_layout.addWidget(btn_remove)
        
        group_layout.addLayout(header_layout)
        
        # Experiment checkboxes
        exp_layout = QVBoxLayout()
        exp_checkboxes = []
        
        for i, exp in enumerate(self.experiments):
            checkbox = QCheckBox(get_experiment_display_name(exp, i))
            exp_layout.addWidget(checkbox)
            exp_checkboxes.append((checkbox, i))
        
        group_layout.addLayout(exp_layout)
        
        self.groups_layout.addWidget(group_frame)
        self.groups.append({
            'frame': group_frame,
            'name_edit': name_edit,
            'color_btn': color_btn,
            'checkboxes': exp_checkboxes
        })
    
    def _remove_group(self, frame):
        """Remove a group widget."""
        for i, group in enumerate(self.groups):
            if group['frame'] == frame:
                self.groups_layout.removeWidget(frame)
                frame.deleteLater()
                self.groups.pop(i)
                break
    
    def get_selected_timepoint_index(self):
        """Return the file index for the selected timepoint."""
        return self.timepoint_combo.currentData()
    
    def get_selected_timepoint_label(self):
        """Return the label of the selected timepoint."""
        return self.timepoint_combo.currentText()
    
    def get_groups(self):
        """Return list of groups with their experiments and colors."""
        result = []
        for group in self.groups:
            selected_exps = [exp_idx for checkbox, exp_idx in group['checkboxes'] if checkbox.isChecked()]
            if selected_exps:
                result.append({
                    'name': group['name_edit'].text(),
                    'color': group['color_btn'].get_color(),
                    'experiments': selected_exps
                })
        return result

    def _save_settings(self):
        """Save current dialog settings to QSettings for persistence."""
        settings = QSettings("NeuroStemVolt", "GroupedAmplitudeDialog")
        
        # Save timepoint index
        settings.setValue("timepoint_index", self.timepoint_combo.currentIndex())
        
        # Save groups info as JSON
        groups_data = []
        for group in self.groups:
            group_info = {
                'name': group['name_edit'].text(),
                'color': group['color_btn'].get_color(),
                'checked_experiments': [exp_idx for checkbox, exp_idx in group['checkboxes'] if checkbox.isChecked()]
            }
            groups_data.append(group_info)
        
        settings.setValue("groups_data", json.dumps(groups_data))
    
    def _load_settings(self):
        """Load saved settings from QSettings."""
        settings = QSettings("NeuroStemVolt", "GroupedAmplitudeDialog")
        
        # Load timepoint index
        tp_index = settings.value("timepoint_index", None)
        if tp_index is not None:
            idx = int(tp_index)
            if 0 <= idx < self.timepoint_combo.count():
                self.timepoint_combo.setCurrentIndex(idx)
        
        # Load groups data
        groups_json = settings.value("groups_data", None)
        if groups_json:
            try:
                groups_data = json.loads(groups_json)
                
                # Remove default groups and recreate from saved data
                while self.groups:
                    self._remove_group(self.groups[0]['frame'])
                
                for group_info in groups_data:
                    self._add_group()
                    group = self.groups[-1]
                    
                    # Restore name
                    group['name_edit'].setText(group_info.get('name', f"Group {len(self.groups)}"))
                    
                    # Restore color
                    group['color_btn'].set_color(group_info.get('color', self.DEFAULT_COLORS[len(self.groups) - 1]))
                    
                    # Restore checked experiments
                    checked_exps = set(group_info.get('checked_experiments', []))
                    for checkbox, exp_idx in group['checkboxes']:
                        checkbox.setChecked(exp_idx in checked_exps)
                        
            except Exception as e:
                print(f"Warning: Could not load saved settings: {e}")
    
    def accept(self):
        """Override accept to save settings before closing."""
        self._save_settings()
        super().accept()


# Keep the old name as an alias for backward compatibility
GroupedMultiTimepointDialog = GroupedAmplitudeDialog


class ViolinPlotDialog(QDialog):
    """
    Dialog for creating violin/swarm plots comparing differences between two timepoints.
    
    Allows the user to:
    1. Select exactly two timepoints
    2. Group experiments together
    3. Choose colors for each group
    4. Set custom title for the plot
    """
    
    DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def __init__(self, experiments, time_between_files, files_before_treatment, 
                 parent=None, title="Amplitude Distribution Comparison"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(550)
        self.setMinimumHeight(600)
        
        self.experiments = experiments
        self.time_between_files = time_between_files
        self.files_before_treatment = files_before_treatment
        self.groups = []
        
        layout = QVBoxLayout(self)
        
        # Plot title input
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Plot Title:"))
        self.plot_title_edit = QLineEdit("Amplitude Distribution Comparison")
        title_layout.addWidget(self.plot_title_edit)
        layout.addLayout(title_layout)
        
        # Y-axis label input
        ylabel_layout = QHBoxLayout()
        ylabel_layout.addWidget(QLabel("Y-axis Label:"))
        self.ylabel_edit = QLineEdit("Amplitude (nM)")
        ylabel_layout.addWidget(self.ylabel_edit)
        layout.addLayout(ylabel_layout)
        
        # Timepoints selection (exactly two)
        layout.addWidget(QLabel("Select exactly TWO timepoints:"))
        
        tp_layout = QHBoxLayout()
        tp_layout.addWidget(QLabel("Timepoint 1:"))
        self.timepoint1_combo = QComboBox()
        tp_layout.addWidget(self.timepoint1_combo)
        
        tp_layout.addWidget(QLabel("Timepoint 2:"))
        self.timepoint2_combo = QComboBox()
        tp_layout.addWidget(self.timepoint2_combo)
        layout.addLayout(tp_layout)
        
        # Populate timepoint combos
        self._populate_timepoints()
        
        # Groups section
        layout.addWidget(QLabel("Create groups and select experiments for each:"))
        
        # Scrollable area for groups
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        self.groups_container = QWidget()
        self.groups_layout = QVBoxLayout(self.groups_container)
        self.groups_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll.setWidget(self.groups_container)
        layout.addWidget(scroll, stretch=1)
        
        # Add group button
        group_btn_layout = QHBoxLayout()
        btn_add_group = QPushButton("+ Add Group")
        btn_add_group.clicked.connect(self._add_group)
        group_btn_layout.addWidget(btn_add_group)
        group_btn_layout.addStretch()
        layout.addLayout(group_btn_layout)
        
        # Add initial two groups
        self._add_group()
        self._add_group()
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _populate_timepoints(self):
        """Populate timepoint combos."""
        if not self.experiments:
            return
        
        n_files = self.experiments[0].get_file_count()
        for i in range(n_files):
            time_min = (i - self.files_before_treatment) * self.time_between_files
            label = f"{time_min:.0f} min"
            self.timepoint1_combo.addItem(label, i)
            self.timepoint2_combo.addItem(label, i)
        
        # Set second combo to different value if possible
        if n_files > 1:
            self.timepoint2_combo.setCurrentIndex(1)
    
    def _add_group(self):
        """Add a new group widget."""
        group_idx = len(self.groups)
        
        group_frame = QFrame()
        group_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        group_frame.setStyleSheet("QFrame { background-color: #f5f5f5; border-radius: 5px; }")
        
        group_layout = QVBoxLayout(group_frame)
        group_layout.setContentsMargins(10, 10, 10, 10)
        
        # Group header
        header_layout = QHBoxLayout()
        
        name_edit = QLineEdit(f"Group {group_idx + 1}")
        name_edit.setPlaceholderText("Group name (e.g., 'male', 'female')")
        name_edit.setMaximumWidth(200)
        
        color_btn = ColorButton(self.DEFAULT_COLORS[group_idx % len(self.DEFAULT_COLORS)])
        
        btn_remove = QPushButton("×")
        btn_remove.setFixedSize(25, 25)
        btn_remove.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        btn_remove.clicked.connect(lambda checked, f=group_frame: self._remove_group(f))
        
        header_layout.addWidget(QLabel("Name:"))
        header_layout.addWidget(name_edit)
        header_layout.addWidget(QLabel("Color:"))
        header_layout.addWidget(color_btn)
        header_layout.addStretch()
        header_layout.addWidget(btn_remove)
        
        group_layout.addLayout(header_layout)
        
        # Experiment checkboxes
        exp_layout = QVBoxLayout()
        exp_checkboxes = []
        
        for i, exp in enumerate(self.experiments):
            checkbox = QCheckBox(get_experiment_display_name(exp, i))
            exp_layout.addWidget(checkbox)
            exp_checkboxes.append((checkbox, i))
        
        group_layout.addLayout(exp_layout)
        
        self.groups_layout.addWidget(group_frame)
        self.groups.append({
            'frame': group_frame,
            'name_edit': name_edit,
            'color_btn': color_btn,
            'checkboxes': exp_checkboxes
        })
    
    def _remove_group(self, frame):
        """Remove a group widget."""
        for i, group in enumerate(self.groups):
            if group['frame'] == frame:
                self.groups_layout.removeWidget(frame)
                frame.deleteLater()
                self.groups.pop(i)
                break
    
    def get_plot_title(self):
        """Return the custom plot title."""
        return self.plot_title_edit.text()
    
    def get_ylabel(self):
        """Return the custom y-axis label."""
        return self.ylabel_edit.text()
    
    def get_selected_timepoints(self):
        """Return the two selected timepoints as [(idx1, label1), (idx2, label2)]."""
        return [
            (self.timepoint1_combo.currentData(), self.timepoint1_combo.currentText()),
            (self.timepoint2_combo.currentData(), self.timepoint2_combo.currentText())
        ]
    
    def get_groups(self):
        """Return list of groups with their experiments and colors."""
        result = []
        for group in self.groups:
            selected_exps = [exp_idx for checkbox, exp_idx in group['checkboxes'] if checkbox.isChecked()]
            if selected_exps:
                result.append({
                    'name': group['name_edit'].text(),
                    'color': group['color_btn'].get_color(),
                    'experiments': selected_exps
                })
        return result
