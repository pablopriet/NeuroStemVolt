from .base import Processor

class Normalize():
    def process(self, data, context=None):
        """
        Usage:
        This processor normalizes the input data by dividing each scan by the maximum value of that scan.
        """
        if context is not None and  context["experiment_first_peak"] is not None:
            print(f"Experiment first peak: {context['experiment_first_peak']}")
            # Normalize each scan by the first peak value
            normalized_data = data / context["experiment_first_peak"]
        else:
            print("No experiment first peak found in context, normalization cannot proceed")
        return normalized_data