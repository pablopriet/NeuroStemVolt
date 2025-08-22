from core.spheroid_file import SpheroidFile
from core.pipeline_manager import PipelineManager
from core.processing.raw_mean import RawMean
from core.processing.rolling_mean import RollingMean
from core.processing.butterworth import ButterworthFilter
from core.processing.gaussian import GaussianSmoothing2D
from core.processing.baseline_correct import BaselineCorrection
from core.processing.find_amplitude import FindAmplitude
from core.processing.normalize import Normalize
from core.processing.sav_gol import SavitzkyGolayFilter
from core.processing.background_subtraction import BackgroundSubtraction
from core.processing.exponentialdecay import ExponentialFitting
from core.utils import extract_timepoint
import os

class SpheroidExperiment:
    """
    Represents a full experimental (a single replicate) dataset for a single spheroid.

    This class handles the initialization, preprocessing, and analysis of multiple
    time-resolved recordings (files) representing the same spheroid across time or treatment.

    Args:
        filepaths (list): List of paths to data files.
        file_length (int, optional): Duration of each file in seconds. Default is 60.
        acquisition_frequency (int, optional): Data acquisition rate in Hz. Default is 10.
        peak_position (int, optional): Index of peak amplitude. Default is 257.
        treatment (str, optional): Description of treatment applied. Default is "".
        waveform (str, optional): Label for waveform (e.g., "5HT"). Default is "5HT".
        stim_params (dict, optional): Dictionary of stimulation settings. Defaults set if None.
        processors (list, optional): List of Processor objects. Default is standard pipeline.
        time_between_files (float, optional): Interval between recordings in minutes. Default is 10.0.
        files_before_treatment (int, optional): Number of baseline recordings. Default is 3.
        file_type (optional): Type or label for input file source.

    Attributes:
        files (list): List of SpheroidFile instances.
        processors (list): Processing pipeline steps.
        stim_params (dict): Dictionary of stimulation metadata.
    """
    def __init__(
        self,
        filepaths,
        file_length=60,
        acquisition_frequency=10, 
        peak_position=257,
        treatment="",
        waveform="",
        stim_params=None,
        processors=None,
        time_between_files= 10.0,
        files_before_treatment = 3,
        file_type = None
    ):
        if waveform is None:
            self.waveform = "5HT"
        else:
            self.waveform = waveform

        self.acquisition_frequency = acquisition_frequency
        self.files = [SpheroidFile(fp,  self.acquisition_frequency, self.waveform) for fp in sorted(filepaths, key=extract_timepoint)]
        self.file_length = file_length
        self.peak_position = peak_position
        self.treatment = treatment
        self.time_between_files = time_between_files
        self.files_before_treatment = files_before_treatment
        self.file_type = file_type

        # Initialize processors after acquisition_frequency is set
        if processors is None:
            # Choose the appropriate amplitude finder based on file type
            if file_type == "Spontaneous":
                from core.processing.spontaneous_peak_detector import FindAmplitudeMultiple
                amplitude_processor = FindAmplitudeMultiple(self.peak_position)
            else:
                amplitude_processor = FindAmplitude(self.peak_position)
            
            self.processors = [
                BackgroundSubtraction(region=(0, 10)),
                ButterworthFilter(),
                BaselineCorrection(),
                Normalize(self.peak_position),
                amplitude_processor,  # Use the appropriate processor
                ExponentialFitting(),
            ]
        else:
            self.processors = processors

        # Default stimulation parameters if not provided
        self.stim_params = stim_params or {
            "start": 5.0,
            "duration": 2.0,
            "frequency": 20,
            "amplitude": 0.5,
            "pulses": 50
        }

        self.set_peak_position(peak_position)

    def set_peak_position(self, peak_position):
        """
        Set peak detection index for all files in this experiment.

        Args:
            peak_position (int): Sample index of expected peak.

        Returns:
            None
        """
        for f in self.files:
            f.set_peak_position(peak_position)

    def get_file_count(self):
        """Return number of files in the experiment."""
        return len(self.files)
    
    def get_file_length(self):
        """Return the duration of each file in seconds."""
        return self.file_length
    
    def get_acquisition_frequency(self):
        """Return the acquisition frequency in Hz."""
        return self.acquisition_frequency
    
    def get_file_time_points(self):
        """Return number of time points per file."""
        return int(self.file_length) * int(self.acquisition_frequency)
    
    def get_number_of_files_before_treatment(self):
        """Return the number of baseline files before treatment."""
        return self.files_before_treatment

    def get_spheroid_file(self, index):
        """
        Return a specific SpheroidFile by index.

        Args:
            index (int): Index of desired file.

        Returns:
            SpheroidFile: The requested file.

        Raises:
            IndexError: If index is out of range.
        """
        if 0 <= index < len(self.files):
            return self.files[index]
        else:
            raise IndexError("Spheroid file index out of range")

    def get_time_between_files(self):
        """Return time interval between files in minutes."""
        return self.time_between_files
    
    def revert_processing(self):
        """
        Revert all processed data back to its original form.

        Returns:
            None
        """
        for spheroid_file in self.files:
            spheroid_file.set_processed_data_as_original()

    def set_processing_steps(self, processors = None):
        """
        Set the processing steps (pipeline).

        Args:
            processors (list): List of Processor objects.

        Returns:
            None
        """
        self.processors = processors

    def get_processing_steps(self):
        """Return the list of configured processing steps."""
        return self.processors

    def run(self):
        """
        Run the entire processing pipeline across all files.

        Returns:
            None
        """
        pipeline = PipelineManager(self.processors)
        # Add stimulation parameters to the context
        context = {
            "peak_position": self.peak_position,
            "stim_start": self.stim_params.get("start", 0),
            "stim_duration": self.stim_params.get("duration", 0),
            "stim_frequency": self.stim_params.get("frequency", 0),
            "acquisition_frequency": self.acquisition_frequency,
        }
        for spheroid_file in self.files:
            pipeline.run(spheroid_file, context=context)

    def run_single_file(self, index):
        """
        Run the processing pipeline for one specific file.

        Args:
            index (int): Index of file to process.

        Returns:
            None
        """
        pipeline = PipelineManager(self.processors)
        pipeline.run(self.files[index])

if __name__ == "__main__":
    #folder = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241111_batch1_n1_Sert"
    #folder = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241111_batch1_n1_Sert"
    folder = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241116_batch1_n3_Sert"
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]

    experiment = SpheroidExperiment(filepaths, treatment="Sertraline", waveform = "5HT")
    experiment.run()
    print(f"Number of files (timepoints) in this experiment: {experiment.get_file_count()}")
    print(f"First file used for baseline: {experiment.get_spheroid_file(3).get_filepath()}")
    # Access the metadata of the first SpheroidFile
    spheroid_file = experiment.get_spheroid_file(0)
    print(spheroid_file.metadata)
    experiment.get_spheroid_file(15).visualize_color_plot_data(title_suffix="Baseline")
    experiment.get_spheroid_file(15).visualize_cv()
    experiment.get_spheroid_file(15).visualize_3d_color_plot(title_suffix="Raw Data")
    #experiment.get_spheroid_file(15).animate_3d_color_plot(title_suffix="Processed Data")
    experiment.get_spheroid_file(15).visualize_IT_profile()
    experiment.get_spheroid_file(15).visualize_IT_with_exponential_decay()
    metadata = experiment.get_spheroid_file(1).get_metadata()
    print(metadata.keys(), metadata.values())