import numpy as np
from .base import Processor


class StimArtifactRemoval(Processor):
    """
    Removes the stimulation artifact from FSCV colorplot data by linear interpolation.

    Operates on the full 2D (time_points × voltage_steps) array. Auto-detects the
    artifact by finding scan-to-scan jumps that exceed a MAD-based threshold, then
    interpolates linearly across each detected artifact region for every voltage column.

    Only applied to files where context['has_artifact'] is True; otherwise the data
    is returned unchanged.

    Args:
        threshold (float): MAD multiplier for jump detection. Higher = less sensitive.
        pad (int): Extra scans to include on each side of a detected artifact region.
        max_artifact_scans (int | None): Maximum artifact window in scans. Defaults to
            2 × acquisition_frequency when None.

    Context keys used:
        has_artifact (bool): Whether this file contains a stimulation artifact.
        acquisition_frequency (float): Samples per second.
    """

    def __init__(self, threshold=8, pad=2, max_artifact_scans=None):
        self.threshold = threshold
        self.pad = pad
        self.max_artifact_scans = max_artifact_scans

    def process(self, data, context=None):
        context = context or {}

        # Change this to see the artifact removal works
        debug=False
        self.debug = debug

        acq_freq = context.get('acquisition_frequency', 10)
        n_time, n_voltages = data.shape

        if self.debug:
            print(f"[ArtifactRemoval] Running on data shape=({n_time}, {n_voltages}), "
                  f"acq_freq={acq_freq}, threshold={self.threshold}, pad={self.pad}, "
                  f"max_artifact_scans={self.max_artifact_scans}")

        # For consecutive scans, compute the difference, because you are subtracting the first one against the second, 
        # you are effectively removing one scan, so the shape shrinks a little bit
        scan_to_scan = np.diff(data, axis=0)  # shape: (n_time-1, n_voltages)
        # Compute RMS, if there has been a huge jump somethign dramatically changed
        # If there is a big change, a large value then something dramatically shifted across the whole sweep, i.e. an artifact.
        # The size of the RMS, will be 600x1, one number per scan transition
        jump_magnitude = np.sqrt(np.mean(scan_to_scan ** 2, axis=1))

        # Find the median across the scan transition to find what the typical transition will be 
        median_jump = np.median(jump_magnitude)
        # Median Absolute Deviation to find what the typical deviation transition will be 
        mad_jump = np.median(np.abs(jump_magnitude - median_jump))
        if mad_jump == 0:
            mad_jump = 1e-12
        # Note that we use median as they are resistant to outliers and will not be inflated like the mean

        threshold_factor = self.threshold
        pad = self.pad
        max_artifact_length = self.max_artifact_scans if self.max_artifact_scans is not None else int(acq_freq * 2)

        # Now that we have found the median, find the threshold given the median threshold factor (how many times the MAD) 
        detection_threshold = median_jump + threshold_factor * mad_jump

        if self.debug:
            print(f"[ArtifactRemoval] Jump stats — median={median_jump:.4f}, MAD={mad_jump:.4f}, "
                  f"threshold={detection_threshold:.4f}, max_artifact_length={max_artifact_length} scans")

        # Finally find the indices where the big jump occurs
        # from the RMS in comparison with the detection threshold (both are in nA units so its an okay comparison)
        jump_mask = jump_magnitude > detection_threshold
        jump_indices = np.where(jump_mask)[0]

        if self.debug:
            print(f"[ArtifactRemoval] Detected {len(jump_indices)} jump(s) at scan indices: {jump_indices.tolist()}")

        if len(jump_indices) == 0:
            if self.debug:
                print("[ArtifactRemoval] No jumps found — returning data unchanged.")
            return data

        # Make a dummy mask of regions where the artifact can be 
        # Cluster nearby jump points into artifact regions
        artifact_mask = np.zeros(n_time, dtype=bool)

        i = 0
        while i < len(jump_indices):
            onset = jump_indices[i] + 1  # first affected scan 
            # Adding +1 to the index after scan

            # Look for the matching offset jump nearby, this is because
            # whenever we look for jumps, there can be one onset and one back to baseline
            # This is so we dont process this as separate jumps, otherwise single scan is corrupt
            offset = onset  # at minimum this one scan is bad
            for j in range(i + 1, len(jump_indices)):
                if jump_indices[j] - jump_indices[i] <= max_artifact_length:
                    offset = jump_indices[j]  # last affected scan
                    i = j  # skip past paired jumps
                    break

            # Mark affected scans with padding, 
            # artifacts may not be perfectly isolated, 
            # so pad before and after
            start = max(0, onset - pad)
            end = min(n_time, offset + pad + 1)
            artifact_mask[start:end] = True
            i += 1

        if not np.any(artifact_mask):
            if self.debug:
                print("[ArtifactRemoval] Artifact mask is empty after clustering — returning data unchanged.")
            return data

        # Find contiguous artifact regions 
        edges = np.diff(artifact_mask.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1

        if artifact_mask[0]:
            starts = np.concatenate(([0], starts))
        if artifact_mask[-1]:
            ends = np.concatenate((ends, [n_time]))

        if self.debug:
            regions = list(zip(starts.tolist(), ends.tolist()))
            print(f"[ArtifactRemoval] {len(regions)} artifact region(s) to interpolate: {regions}")

        # Interpolate!
        result = data.copy()

        for start_idx, end_idx in zip(starts, ends):
            if start_idx < 1 or end_idx >= n_time:
                if self.debug:
                    print(f"[ArtifactRemoval]   Skipping region [{start_idx}:{end_idx}] — at data boundary.")
                continue

            if self.debug:
                print(f"[ArtifactRemoval]   Interpolating scans [{start_idx}:{end_idx}] "
                      f"({end_idx - start_idx} scans, {(end_idx - start_idx) / acq_freq:.2f}s)")

            x_interp = np.arange(start_idx, end_idx)
            xp = [start_idx - 1, end_idx]

            for v in range(n_voltages):
                fp = [result[xp[0], v], result[xp[1], v]]
                result[start_idx:end_idx, v] = np.interp(x_interp, xp, fp)

        if self.debug:
            print("[ArtifactRemoval] Done.")

        return result
