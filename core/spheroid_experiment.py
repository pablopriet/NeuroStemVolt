from spheroid_file import SpheroidFile
from pipeline_manager import PipelineManager
from processing.raw_mean import RawMean
from processing.rolling_mean import RollingMean
from processing.butterworth import ButterworthFilter
from processing.gaussian import GaussianSmoothing2D
from processing.baseline_correct import BaselineCorrection
from processing.find_amplitude import FindAmplitude
from processing.sav_gol import SavitzkyGolayFilter
from processing.background_subtraction import BackgroundSubtraction
from group_analysis import GroupAnalysis
from output_manager import OutputManager
from utils import extract_timepoint
import os

class SpheroidExperiment:
    """
    Represents one full experiment for a single spheroid.
    It contains multiple SpheroidFile objects, one per timepoint.
    """

    def __init__(
        self,
        filepaths,
        file_length=60,
        acquisition_frequency=10,
        peak_position=257,
        treatment="Sertraline",
        stim_params=None,
        processors=None  # Default to None
    ):
        self.files = [SpheroidFile(fp) for fp in sorted(filepaths, key=extract_timepoint)]
        self.file_length = file_length
        self.acquisition_frequency = acquisition_frequency
        self.peak_position = peak_position
        self.treatment = treatment

        # Initialize processors after acquisition_frequency is set
        if processors is None:
            self.processors = [
                BackgroundSubtraction(region=(0, 10)),
                #SavitzkyGolayFilter(w=20, p=2),
                ButterworthFilter(),
                FindAmplitude(self.peak_position)
                # No need to pass context here
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
        for f in self.files:
            f.set_peak_position(peak_position)

    def get_file_count(self):
        return len(self.files)

    def get_spheroid_file(self, index):
        if 0 <= index < len(self.files):
            return self.files[index]
        else:
            raise IndexError("Spheroid file index out of range")

    def run(self):
        """
        Runs the processing pipeline across all files.
        """
        pipeline = PipelineManager(self.processors)
        for spheroid_file in self.files:
            pipeline.run(spheroid_file)

    def run_single_file(self, index):
        """
        Runs the processing pipeline for a single file specified by index.
        """
        pipeline = PipelineManager(self.processors)
        pipeline.run(self.files[index])

    def collect(self):
        """
        Collects output results (future implementation).
        """
        pass


if __name__ == "__main__":
    #folder = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241111_batch1_n1_Sert"
    folder = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241111_batch1_n1_Sert"
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]

    experiment = SpheroidExperiment(filepaths, treatment="Sertraline")
    experiment.run()
    print(f"Number of files (timepoints) in this experiment: {experiment.get_file_count()}")
    print(f"First file used for baseline: {experiment.get_spheroid_file(3).get_filepath()}")
    experiment.get_spheroid_file(1).visualize_color_plot_data(title_suffix="Baseline")
    experiment.get_spheroid_file(1).visualize_IT_profile()

