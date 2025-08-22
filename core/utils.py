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
