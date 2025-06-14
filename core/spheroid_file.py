import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class SpheroidFile:
    """
    A class to handle the loading and processing of FSCV spheroid data files.
    Usage:
    - filepath: Path to the associated FSCV (color plot) data file (txt format).
    - raw_data: The raw data loaded from the file, inverted. We keep it to always come back in case the user does not want to apply any processing.
    - processed_data: The processed data, initially set to raw_data.
    - peak_position: The position of the peak in the I-T profile, default is set to 257 (for serotonin, 5HT).
    - window_size: The size of the window for rolling mean or smoothing, default is None. (not used yet)
    - metadata: A dictionary to hold any additional metadata related to the file.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = self.load_data()
        self.processed_data = self.raw_data
        self.peak_position = 257 # Default peak position for serotonin (5HT) in FSCV plots
        self.window_size = None # Default window size for rolling mean or smoothing
        self.metadata = {}

    def load_data(self):
        initial_data = np.loadtxt(self.filepath).T # Transpose to match (voltage steps, time points)
        data = -initial_data # Invert the data since FSCV txt color plots will be inverted
        return data
    
    def set_peak_position(self, peak_position):
        self.peak_position = peak_position
    
    def get_data(self):
        return self.raw_data
        
    def get_filepath(self):
        return self.filepath
        
    def update_metadata(self, context):
        """
        Updates the metadata with the provided context.
        """
        self.metadata.update(context)

    def get_metadata(self):
        return self.metadata
    
    def visualize_color_plot_data(self, title_suffix=""):
        # Initialize plot settings
        plot_settings = PLOT_SETTINGS()
        custom_cmap = plot_settings.custom
        
        # Use the loaded data
        processed_data = self.processed_data
        
        # Calculate limits
        vmin = np.percentile(processed_data, 1)
        vmax = np.percentile(processed_data, 99)

        # Create and save plot
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(processed_data.T, # Transpose to match time points and voltage steps, imshow expects (y, x)
                      aspect='auto', 
                      cmap=custom_cmap, 
                      origin='lower',
                      extent = [0, processed_data.shape[1], 0, processed_data.shape[0]],
                      vmin=vmin, 
                      vmax=vmax)
        plt.colorbar(im, ax=ax, label="Current (nA)")
        ax.set_xlabel("Time Points")
        ax.set_ylabel("Voltage Steps")
        ax.set_title(f"Color Plot{': ' + title_suffix if title_suffix else ''}\nRange: [{vmin:.2f}, {vmax:.2f}] nA")
        
        plt.tight_layout()
        plt.show()

        return fig, ax
    
    def visualize_IT_profile(self):
        """
        Visualizes the I-T profile at the specified peak position and highlights all detected peaks.
        """
        # Extract the profile at the peak position
        profile = self.processed_data[:, self.peak_position]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(profile, color='blue', linewidth=1.5, label="I-T Profile")

        # Highlight all peak positions if available in metadata
        if 'peak_amplitude_positions' in self.metadata:
            peak_indices = self.metadata['peak_amplitude_positions']
            peak_values = self.metadata.get('peak_amplitude_values', profile[peak_indices])
            print(f"Peak indices from metadata: {peak_indices}")
            for peak_idx, peak_val in zip(peak_indices, peak_values): # Iterate over index-value pairs
                if 0 <= peak_idx < len(profile): # Check the peak is within bounds
                    ax.scatter(peak_idx, peak_val, color='red', label="Peak Amplitude", zorder=5)
                    ax.annotate(f"({peak_idx}, {peak_val:.2f})",
                                (peak_idx, peak_val),
                                textcoords="offset points",
                                xytext=(10, 10),
                                ha='center',
                                fontsize=10,
                                color='red')

        # Add labels, title, and grid
        ax.set_xlabel("Time Points")
        ax.set_ylabel("Current (nA)")
        ax.set_title(f"I-T Profile at Peak Position {self.peak_position}")
        ax.grid(False)
        ax.legend()

        # Show the plot
        plt.show()

    def visualize_IT_with_exponential_decay(self):
        """
        Visualizes the I-T profile, highlights the detected peak, overlays the exponential decay curve,
        and marks the half-life (t_half).
        """
        # Extract the I-T profile at the specified peak position
        profile = self.processed_data[:, self.peak_position]

        # Extract the exponential decay parameters from metadata
        if "exponential fitting parameters" not in self.metadata:
            raise ValueError("Exponential fitting parameters are missing from metadata.")
        
        params = self.metadata["exponential fitting parameters"]
        A = params["A"]
        tau = params["tau"]
        C = params["C"]
        t_half = params["t_half"]  # Directly retrieve t_half from params

        # Extract the peak position from metadata
        if "peak_amplitude_positions" not in self.metadata:
            raise ValueError("Peak amplitude positions are missing from metadata.")
        
        peak_positions = self.metadata["peak_amplitude_positions"]
        if len(peak_positions) == 0:
            raise ValueError("No peak positions found in metadata.")
        
        peak_position = int(peak_positions[0])  # Use the first peak position

        # Slice the profile starting from the peak position
        y = profile[peak_position:]
        t = np.arange(peak_position, peak_position + len(y))

        # Generate the exponential decay curve
        exp_decay_curve = A * np.exp(-t / tau) + C

        # Calculate the corresponding position and value for t_half
        t_half_position = peak_position + int(t_half)  # Convert t_half to the corresponding position
        t_half_value = A * np.exp(-t_half / tau) + C  # Value of the exponential decay at t_half

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(profile, color='blue', linewidth=1.5, label="I-T Profile")
        ax.scatter(peak_position, profile[peak_position], color='red', label="Detected Peak", zorder=5)
        ax.plot(t, exp_decay_curve, color='green', linestyle='--', label="Exponential Decay Fit")

        # Plot the half-life (t_half)
        ax.axvline(x=t_half_position, color='purple', linestyle=':', label=f"t_half = {t_half:.2f}")

        # Add labels, title, and legend
        ax.set_xlabel("Time Points")
        ax.set_ylabel("Current (nA)")
        ax.set_title(f"I-T Profile with Exponential Decay Fit and Half-Life")
        ax.legend()
        ax.grid(True)

        # Show the plot
        plt.show()

class PLOT_SETTINGS:
    def __init__(self):
        # Custom colormap only
        self.custom = self.get_continuous_cmap(
            ['#001524','#002f5e','#f4c300','#a84900','#64005f','#00a37a','#00751c','#00ff00'],
            [0, 0.2478, 0.3805, 0.6555, 0.701, 0.7603, 0.7779, 1]
        )

    def get_continuous_cmap(self, hex_list, float_list=None):
        rgb_list = [self.rgb_to_dec(self.hex_to_rgb(i)) for i in hex_list]
        if float_list is None:
            float_list = list(np.linspace(0, 1, len(rgb_list)))

        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp

    def hex_to_rgb(self, value):
        value = value.strip("#")
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def rgb_to_dec(self, value):
        return [v/256 for v in value]

if __name__ == "__main__":
    Spheroid_test = SpheroidFile(r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241111_batch1_n1_Sert\04_-30_COLOR.txt")
    Spheroid_test.visualize_color_plot_data("")
    plt.show()
    Spheroid_test.visualize_IT_profile()
    plt.show()
