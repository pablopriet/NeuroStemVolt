# utils/peaks.py
def meta_get_peaks_and_active(metadata: dict):
    """Return (peaks_list, active_idx) in normalized form."""
    md = metadata or {}
    peaks = md.get("peak_amplitude_positions", []) or []
    if hasattr(peaks, "tolist"):  # numpy arrays
        peaks = peaks.tolist()
    if isinstance(peaks, (int, float)):
        peaks = [int(peaks)]
    else:
        try:
            peaks = [int(p) for p in list(peaks)]
        except Exception:
            peaks = []
    active = int(md.get("active_peak_index", 0))
    if not peaks:
        active = 0
    else:
        active = max(0, min(active, len(peaks) - 1))
    return peaks, active


def meta_set_peaks_and_active(file_obj, peaks, active_idx):
    """
    Save peaks and active index AND compute `peak_amplitude_values` from processed data.
    Also keeps convenience fields consistent (primary_* and num_peaks_detected).
    """
    # normalise & de-duplicate peaks, preserving order
    try:
        peaks = [int(p) for p in list(peaks)]
    except Exception:
        peaks = []
    seen = set(); uniq = []
    for p in peaks:
        if p not in seen:
            uniq.append(p); seen.add(p)
    peaks = uniq

    md = (file_obj.get_metadata() or {}).copy()

    # clamp active index
    if peaks:
        try:
            active_idx = int(active_idx)
        except Exception:
            active_idx = 0
        active_idx = max(0, min(active_idx, len(peaks) - 1))
    else:
        active_idx = 0

    # compute values at peaks using current peak_position
    values = _compute_peak_values(file_obj, peaks)

    # write back
    # Single-peak → scalar for backward compat (Stimulation / downstream consumers)
    # Multi-peak or empty → list (Spontaneous mode)
    if len(peaks) == 1:
        md["peak_amplitude_positions"] = peaks[0]
        md["peak_amplitude_values"] = values[0] if values else 0.0
    else:
        md["peak_amplitude_positions"] = peaks
        md["peak_amplitude_values"] = values
    md["active_peak_index"] = active_idx

    # keep convenience fields aligned to active peak
    if peaks:
        md["primary_peak_position"] = int(peaks[active_idx])
        md["primary_peak_value"] = float(values[active_idx]) if values else 0.0
        md["num_peaks_detected"] = len(peaks)
    else:
        md["primary_peak_position"] = None
        md["primary_peak_value"] = 0.0
        md["num_peaks_detected"] = 0

    file_obj.update_metadata(md)

def _compute_peak_values(file_obj, peaks):
    """Return per-peak amplitudes computed the same way as the detector.

    Priority:
      1) If an IT trace is available (shape == [time]), use IT[time_index].
      2) Else, use the colour matrix column at metadata['peak_position']: data[:, peak_pos][time_index].
    """
    try:
        md = file_obj.get_metadata() or {}

        # 1) Prefer a 1-D IT trace if available (already derived at the chosen voltage index)
        get_it = getattr(file_obj, "get_processed_data_IT", None)
        if callable(get_it):
            it = get_it()
            try:
                import numpy as np
                it_arr = np.asarray(it) if it is not None else None
                if it_arr is not None and it_arr.ndim == 1:
                    T = it_arr.shape[0]
                    vals = []
                    for t in peaks:
                        try:
                            ti = int(t)
                        except Exception:
                            ti = 0
                        ti = max(0, min(ti, T - 1))
                        vals.append(float(it_arr[ti]))
                    return vals
            except Exception:
                # fall back to colour matrix path
                pass

        # 2) Fallback: sample from colour matrix column at peak_position
        get_mat = getattr(file_obj, "get_processed_data", None)
        data = get_mat() if callable(get_mat) else None
        if data is None:
            return [0.0 for _ in peaks]

        import numpy as np
        arr = np.asarray(data)
        if arr.ndim < 2:
            return [0.0 for _ in peaks]

        time_len, volt_len = arr.shape[0], arr.shape[1]
        peak_pos = md.get("peak_position")  # voltage index
        if peak_pos is None:
            return [0.0 for _ in peaks]
        try:
            peak_pos = int(peak_pos)
        except Exception:
            return [0.0 for _ in peaks]
        peak_pos = max(0, min(peak_pos, volt_len - 1))

        fx = arr[:, peak_pos]  # detector does fx = data[:, peak_position]
        vals = []
        for t in peaks:
            try:
                ti = int(t)
            except Exception:
                ti = 0
            ti = max(0, min(ti, time_len - 1))
            vals.append(float(fx[ti]))
        return vals

    except Exception:
        # Fail soft with zeros on any unexpected issue
        return [0.0 for _ in peaks]