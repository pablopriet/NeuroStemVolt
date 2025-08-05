import re

def extract_timepoint(filepath):
    """
    Extract a numeric timepoint from a filename.

    This function searches for a number following an underscore in the given filepath
    (e.g., "file_120.txt" → 120). If no such pattern is found, it returns `float('inf')`.

    Args:
        filepath (str): Path or filename string to extract the timepoint from.

    Returns:
        int or float: The extracted timepoint as an integer, or `float('inf')` if not found.
    """
    match = re.search(r"_(\-?\d+)", filepath) 
    return int(match.group(1)) if match else float('inf')

def check_uniform_step_timepoints(filepaths):
    from core.utils import extract_timepoint

    timepoints = [extract_timepoint(fp) for fp in filepaths]
    
    if float('inf') in timepoints:
        raise BadTimepointOrderError("One or more files do not contain valid numeric timepoints.")

    if len(timepoints) < 2:
        return  # Can't infer step size from a single file

    sorted_tp = sorted(timepoints)
    step = sorted_tp[1] - sorted_tp[0]

    expected = [sorted_tp[0] + i * step for i in range(len(sorted_tp))]

    if sorted_tp != expected:
        raise BadTimepointOrderError(
            f"Inconsistent timepoint sequence. Expected uniform step of {step}. Got: {sorted_tp}"
        )

    # OPTIONAL: Check that the user actually *provided* the files in that same order
    if timepoints != sorted_tp:
        raise BadTimepointOrderError(
            f"Files not provided in timepoint order. Got: {timepoints}, expected: {sorted_tp}"
        )
    
class BadTimepointOrderError(Exception):
    """Raised when input filepaths do not follow a uniform timepoint sequence."""
    pass
