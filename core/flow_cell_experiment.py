import os
import numpy as np

from core.spheroid_experiment import SpheroidExperiment
from core.pipeline_manager import PipelineManager


class FlowCellExperiment(SpheroidExperiment):
    """
    FlowCellExperiment extends SpheroidExperiment with helper methods to:
    - identify buffer files (by basename containing 'buffer')
    - compute mean buffer matrix across selected buffer files (with padding)
    - subtract that mean from selected target files
    Buffer subtraction is explicit: call run_buffer_subtraction(...) from the UI.
    """

    def __init__(self, filepaths, injection_start=None, injection_length=None, **kwargs):
        kwargs['file_type'] = "Flow Cell"
        super().__init__(filepaths, **kwargs)

        # detect buffer / concentration files (indices)
        buffer_indices, concentration_indices = self.identify_buffer_indices()
        self.buffer_indices = buffer_indices
        self.concentration_indices = concentration_indices

        self.injection_start = injection_start
        self.injection_length = injection_length
        self.injection_end = None
        if self.injection_start is not None and self.injection_length is not None:
            self.injection_end = self.injection_start + self.injection_length

    def identify_buffer_indices(self):
        """
        Inspect self.files and return (buffer_indices, concentration_indices) as lists of indices.
        Uses the file's filepath basename and checks 'buffer' in the name (case-insensitive).
        Works whether items in self.files are SpheroidFile objects or plain filepaths.
        """
        buffer_indices = []
        concentration_indices = []
        for idx, f in enumerate(self.files):
            # get path string robustly
            path = None
            if isinstance(f, str):
                path = f
            else:
                # prefer public getter if present
                if hasattr(f, "get_filepath"):
                    try:
                        path = f.get_filepath()
                    except Exception:
                        path = getattr(f, "filepath", None)
                else:
                    path = getattr(f, "filepath", None)
            name = os.path.basename(path).lower() if isinstance(path, str) else ""
            if "buffer" in name:
                buffer_indices.append(idx)
            else:
                concentration_indices.append(idx)
        return buffer_indices, concentration_indices

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