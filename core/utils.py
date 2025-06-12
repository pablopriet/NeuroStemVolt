import re

def extract_timepoint(filepath):
    # This function looks for a number preceded by an underscore _
    # The number may be negative (-?) and must contain at least one digit (\d+)
    # If no match is found, it returns float('inf') to indicate no valid timepoint
    # If a match is found, it returns the integer value of the matched number for sorting purposes
    match = re.search(r"_(\-?\d+)", filepath) 
    return int(match.group(1)) if match else float('inf')
