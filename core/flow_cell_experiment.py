import os
import re
import numpy as np

from core.spheroid_experiment import SpheroidExperiment
from core.pipeline_manager import PipelineManager


class FlowCellExperiment(SpheroidExperiment):
    """
    FlowCellExperiment extends SpheroidExperiment with helper methods to:
    - identify buffer files (by basename containing 'buffer' or 'blank')
    - group calibration files by concentration using user‑provided calibration_points
    - compute mean buffer matrix across selected buffer files (with padding)
    - subtract that mean from selected target files
    Buffer subtraction is explicit: call run_buffer_subtraction(...) from the UI.
    """

    def __init__(self, filepaths, injection_start=None, injection_length=None,
                 calibration_points=None, repetitions_per_cal=None, **kwargs):
        kwargs['file_type'] = "Flow Cell"
        super().__init__(filepaths, **kwargs)

        # User‑provided metadata from settings dialog
        # e.g. [500, 250, 100, 50, 25, 10] (nM)
        self.calibration_points = calibration_points or []
        # how many replicates per concentration the user expects
        self.repetitions_per_cal = repetitions_per_cal or 1

        # detect buffer / non‑buffer files (indices)
        buffer_indices, concentration_indices = self.identify_buffer_indices()
        self.buffer_indices = buffer_indices
        self.concentration_indices = concentration_indices

        # Map: concentration_value -> list[file_index]
        self.files_by_concentration = self._organize_by_concentration()

        self.injection_start = injection_start
        self.injection_length = injection_length
        self.injection_end = None
        if self.injection_start is not None and self.injection_length is not None:
            self.injection_end = self.injection_start + self.injection_length

    def identify_buffer_indices(self):
        """
        Inspect self.files and return (buffer_indices, concentration_indices) as lists of indices.
        Uses the file's filepath basename and checks 'buffer' or 'blank' in the name (case-insensitive).
        """
        buffer_indices = []
        concentration_indices = []
        for idx, f in enumerate(self.files):
            path = self._get_filepath(f)
            name = os.path.basename(path).lower() if isinstance(path, str) else ""
            if "buffer" in name or "blank" in name:
                buffer_indices.append(idx)
            else:
                concentration_indices.append(idx)
        return buffer_indices, concentration_indices

    def _get_filepath(self, file_obj):
        """Helper to extract filepath string from SpheroidFile or plain string."""
        if isinstance(file_obj, str):
            return file_obj
        if hasattr(file_obj, "get_filepath"):
            try:
                return file_obj.get_filepath()
            except Exception:
                pass
        return getattr(file_obj, "filepath", None)

    def _parse_conc_from_basename(self, basename):
        """
        Extract numeric concentration (as float) from filename *without* assuming units.
        For flow-cell data like '001_500nM_COLOR.txt', this will yield 500.0.

        We don't convert units here; we only extract the bare number.
        """
        # Look for the first number before 'nm' or in the whole name
        # Examples matched: '500nM', '250nM', '10nM', '075nM', '002-250nM'
        m = re.search(r'(\d+\.?\d*)\s*nm', basename.lower())
        if m:
            return float(m.group(1))

        # Fallback: take first number in the basename
        m2 = re.search(r'(\d+\.?\d*)', basename.lower())
        if m2:
            return float(m2.group(1))

        return None

    def _organize_by_concentration(self):
        """
        Use user‑provided calibration_points to group files by concentration.

        For each non‑buffer file:
          - parse a numeric concentration from the basename
          - match it to one of self.calibration_points (with small tolerance)
          - store file index in files_by_concentration[matched_conc]

        Returns:
            dict: {calibration_conc: [list of file indices]}
        """
        files_by_conc = {float(c): [] for c in (self.calibration_points or [])}

        if not self.calibration_points:
            return files_by_conc

        # simple relative tolerance for matching parsed number to calibration point
        rel_tol = 0.05  # 5%

        for idx in self.concentration_indices:
            f = self.files[idx]
            path = self._get_filepath(f)
            if not path:
                continue
            basename = os.path.basename(path)

            parsed = self._parse_conc_from_basename(basename)
            if parsed is None:
                continue

            # find best matching calibration point
            best_match = None
            best_err = None
            for cal in self.calibration_points:
                cal = float(cal)
                denom = cal if cal != 0 else 1.0
                err = abs(parsed - cal) / denom
                if err <= rel_tol and (best_err is None or err < best_err):
                    best_err = err
                    best_match = cal

            if best_match is not None:
                files_by_conc.setdefault(best_match, []).append(idx)

        return files_by_conc

    def get_sorted_concentrations(self):
        """
        Return list of calibration concentrations present in data, sorted ascending.
        """
        return sorted([c for c, idxs in self.files_by_concentration.items() if idxs])

    def get_files_for_concentration(self, conc):
        """
        Return list of file indices belonging to the given calibration concentration.
        """
        conc = float(conc)
        return self.files_by_concentration.get(conc, [])

    def validate_expected_structure(self):
        """
        Validate that files match the expected calibration structure:
          - each calibration_points value has some files
          - each has (approximately) repetitions_per_cal files
        Returns a small report dict.
        """
        report = {"valid": True, "warnings": [], "errors": []}

        # Check concentrations
        for cal in self.calibration_points or []:
            cal = float(cal)
            idxs = self.files_by_concentration.get(cal, [])
            if not idxs:
                report["warnings"].append(f"No files found for calibration {cal} nM.")
            elif self.repetitions_per_cal and len(idxs) != self.repetitions_per_cal:
                report["warnings"].append(
                    f"Calibration {cal} nM: found {len(idxs)} files, expected {self.repetitions_per_cal}."
                )

        # Buffers
        if not self.buffer_indices:
            report["warnings"].append("No buffer / blank files detected.")

        report["valid"] = not report["errors"]
        return report

    def compute_mean_buffer_array(self, buffer_indices=None, use_processed=False, transpose=True):
        """
        Compute mean matrix across selected buffer files.

        - buffer_indices: list of indices to treat as buffers (defaults to self.buffer_indices)
        - use_processed: if True prefer the processed_data output from SpheroidFile for buffers
        - transpose: if True return mean.T to match your project's convention

        Pads matrices to a common shape (edge padding, fallback to NaN) and uses nanmean.
        """
        idxs = buffer_indices if buffer_indices is not None else self.buffer_indices
        if not idxs:
            return None

        matrices = []
        for i in idxs:
            try:
                sf = self.get_spheroid_file(i)
            except Exception:
                continue

            mat = None
            if use_processed:
                mat = getattr(sf, "processed_data", None)
                if mat is None and hasattr(sf, "get_processed_data"):
                    try:
                        mat = sf.get_processed_data()
                    except Exception:
                        mat = None

            if mat is None:
                # Use raw_data
                if hasattr(sf, "get_data"):
                    try:
                        mat = sf.get_data()
                    except Exception:
                        mat = None
                if mat is None:
                    mat = getattr(sf, "raw_data", None)
                    if mat is None and hasattr(sf, "get_raw_data"):
                        try:
                            mat = sf.get_raw_data()
                        except Exception:
                            mat = None

            if mat is not None:
                matrices.append(np.asarray(mat))

        if not matrices:
            return None

        shapes = [m.shape for m in matrices]
        max_shape = (max(s[0] for s in shapes), max(s[1] for s in shapes))

        # Try edge padding if needed, fallback to constant NaN padding if edge fails
        try:
            matrices_padded = [
                np.pad(m, ((0, max_shape[0] - m.shape[0]), (0, max_shape[1] - m.shape[1])), mode='edge')
                for m in matrices
            ]
        except Exception:
            matrices_padded = [
                np.pad(m, ((0, max_shape[0] - m.shape[0]), (0, max_shape[1] - m.shape[1])), mode='constant', constant_values=np.nan)
                for m in matrices
            ]

        mean_matrix = np.nanmean(matrices_padded, axis=0)

        if transpose:
            mean_matrix = mean_matrix.T

        return mean_matrix

    def subtract_mean_from_targets(self, mean_array, target_indices=None, write_to_processed=True):
        """
        Subtract provided mean_array from target files.
        - If target_indices is None, subtract from all non-buffer files.
        - Writes result into spheroid_file.processed_data (or uses set_processed_data if present).
        Returns True on success (or partial success), False if mean_array is None.
        """
        if mean_array is None:
            return False

        if target_indices is None:
            target_indices = [i for i in range(len(self.files)) if i not in (self.buffer_indices or [])]

        for i in target_indices:
            try:
                sf = self.get_spheroid_file(i)
            except Exception:
                continue

            # obtain source data (prefer raw/original)
            src = None
            if hasattr(sf, "get_data"):
                try:
                    src = sf.get_data()
                except Exception:
                    src = None
            if src is None:
                src = getattr(sf, "raw_data", None)
                if src is None and hasattr(sf, "get_raw_data"):
                    try:
                        src = sf.get_raw_data()
                    except Exception:
                        src = None
            if src is None:
                # fallback to any existing processed data
                src = getattr(sf, "processed_data", None)
            if src is None:
                continue

            src_arr = np.asarray(src)
            try:
                out = src_arr.copy() - mean_array
            except Exception:
                # shape mismatch or incompatible types: skip this file
                continue

            # write back
            if write_to_processed and hasattr(sf, "set_processed_data"):
                try:
                    sf.set_processed_data(out)
                except Exception:
                    setattr(sf, "processed_data", out)
            else:
                setattr(sf, "processed_data", out)

        return True

    def run_buffer_subtraction(self, buffer_indices=None, target_indices=None, use_processed_for_buffers=False, write_to_processed=True):
        """
        Convenience method to compute the mean buffer matrix and subtract from targets.
        This is an explicit operation that should be triggered from the UI (so users control which buffers got used).
        """
        mean_buffer = self.compute_mean_buffer_array(buffer_indices=buffer_indices, use_processed=use_processed_for_buffers)
        success = self.subtract_mean_from_targets(mean_buffer, target_indices=target_indices, write_to_processed=write_to_processed)
        return success

    def run(self):
        """
        Run the usual processing pipeline across all files.
        Buffer subtraction is not performed automatically here; call run_buffer_subtraction() explicitly if desired.
        """
        pipeline = PipelineManager(self.processors)
        # Add flow cell parameters to the context
        context = {
            "peak_position": self.peak_position,
            "injection_start": self.injection_start,
            "injection_duration": self.injection_length,
            "injection_end": self.injection_end,
            "acquisition_frequency": self.acquisition_frequency,
        }
        for spheroid_file in self.files:
            pipeline.run(spheroid_file, context=context)