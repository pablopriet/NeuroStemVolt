import re

def extract_timepoint(filepath):
    """
    Extract a numeric value from a filename for SORTING purposes only.

    This function is used to sort files into chronological order, NOT to determine
    actual experiment timepoints. The actual timepoint labels (e.g., "-30 min", "0 min")
    are calculated using user settings:

        time_min = (file_index - files_before_treatment) × time_between_files

    This means both naming conventions are EQUIVALENT:

    Example with settings: time_between_files=10, files_before_treatment=3

    Convention 1 (explicit times):      Convention 2 (sequential numbers):
    ─────────────────────────────────   ───────────────────────────────────
    -30_COLOR.txt → index 0 → -30 min   01 recording.txt → index 0 → -30 min
    -20_COLOR.txt → index 1 → -20 min   02 recording.txt → index 1 → -20 min
    -10_COLOR.txt → index 2 → -10 min   03 recording.txt → index 2 → -10 min
      0_COLOR.txt → index 3 →   0 min   04 recording.txt → index 3 →   0 min
     10_COLOR.txt → index 4 →  10 min   05 recording.txt → index 4 →  10 min

    Both conventions sort correctly and produce the same timepoint labels!

    Handles multiple naming patterns:
    - Sequence + timepoint format: "04_-30_COLOR.txt" → -30 (extracts SECOND number)
    - Leading negative number: "-30_COLOR.txt" → -30 (for sorting)
    - Underscore-prefixed numbers: "file_-30_COLOR.txt" → -30 (for sorting)
    - Leading numbers: "01 recording.txt" → 1 (for sorting)
    - Generic numbers: "recording_120.txt" → 120 (for sorting)

    Args:
        filepath (str): Path or filename string to extract the sort key from.

    Returns:
        int or float: The extracted number for sorting, or `float('inf')` if not found.
    """
    import os
    filename = os.path.basename(filepath)

    # Pattern 1: Sequence_Timepoint format: "04_-30_COLOR.txt" → -30
    # Matches: digits, underscore, (capture: optional minus + digits), underscore
    # This ensures we get the SECOND number (timepoint) not the first (sequence)
    match = re.search(r"^\d+_(-?\d+)_", filename)
    if match:
        return int(match.group(1))

    # Pattern 2: Leading negative number at start: "-30_COLOR.txt" → -30
    match = re.search(r"^(-?\d+)_", filename)
    if match:
        return int(match.group(1))

    # Pattern 3: Underscore followed by a (possibly negative) number
    # e.g., "file_-30_COLOR.txt" → -30
    match = re.search(r"_(-?\d+)", filename)
    if match:
        return int(match.group(1))

    # Pattern 4: Leading number at start of filename (new format)
    # e.g., "01 recording.txt" → 1, "16_recording.txt" → 16
    match = re.search(r"^(\d+)", filename)
    if match:
        return int(match.group(1))

    # Pattern 5: Any number in the filename as fallback
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))

    return float('inf')
