
# DEFAULT PATHS
DEFAULT_INPUT_PATH = f"/Users/tomas/Developer/NeuroStemVolt/data"
DEFAULT_OUTPUT_PATH = f"/Users/tomas/Developer/NeuroStemVolt/output"

# DEFAULT FSCV PARAMETERS
DEFAULT_FREQUENCY = 10  # Hz
# This is the default position of the peak of serotonin (5HT) in the FSCV plot
DEFAULT_PEAK_POSITION = 1000 #HA waveform, for serotonin 1213, for HA 1000
DEFAULT_FILE_LENGTH = 60  # seconds

# STIMULATION PARAMETERS
DEFAULT_STIMULATION_START = 5  # Stimulation starts at 5 seconds within the file
DEFAULT_STIMULATION_FREQUENCY = 20  # Hz
DEFAULT_STIMULATION_DURATION = 0.5  # seconds
DEFAULT_STIMULATION_AMPLITUDE = 0.5  # Volts
DEFAULT_STIMULATION_PULSES = 50  # Number of pulses
# minutes, this means how often are you taking stimulation files for a single spheroid
DEFAULT_INTERVAL_BETWEEN_FILES = 10

# Waveforms
WF_5HT = [0.2, [1.0, -0.1], 0.2]
WF_HA = [-0.5, [-0.7, 1.1], -0.5]
WF_DA = [-0.4, [1.3, -0.4], -0.4]
