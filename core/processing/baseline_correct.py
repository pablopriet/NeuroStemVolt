from .base import Processor

class BaselineCorrection(Processor):
    def process(self, data, context=None):
        """
        Usage:
        This processor applies baseline correction to the input data by subtracting
        the mean of the FSCV recordings previous to stimulation.
        """
        if context is None or "stim_start" not in context:
            raise ValueError("Stimulation start time ('stim_start') is missing from the context.")

        # Get stimulation start time from the context
        stim_start = context["stim_start"]
        peak_position = context.get("peak_position", 257) # Default set to 5HT

        # Calculate the index corresponding to the stimulation start time
        # data.shape[1] corresponds to the time dimension of the data
        acquisition_frequency = context.get("acquisition_frequency", 10)  # Default to 10 Hz
        stim_start_idx = int(stim_start * acquisition_frequency)
        # Define the baseline region as the data before the stimulation start
        baseline_region = data[:, peak_position] # IT profile
        baseline_region = baseline_region[:stim_start_idx] 
        # Compute the mean of the baseline region across the time dimension
        baseline = baseline_region.mean(axis=0, keepdims=True)
        # Store the baseline in the context for later use
        context["baseline"] = baseline
        print(f"Baseline shape: {baseline}")
        # Subtract the baseline from the entire data
        data -= baseline
    
        return data
