from .base import Processor
import scipy.signal as signal

class SavitzkyGolayFilter(Processor):
    """
    Apply a Savitzky-Golay filter to smooth noisy FSCV data while preserving peak features.

    The Savitzky-Golay filter is a digital filter that fits successive sub-sets of adjacent data points 
    with a low-degree polynomial using linear least squares. Unlike simple moving averages, it can preserve 
    features of the distribution such as peak height and width, making it suitable for analyzing 
    fast-scan cyclic voltammetry (FSCV) data with sharp transient responses.

    Attributes:
        w (int): Window size (number of points); must be odd. Larger windows smooth more but may lose detail.
        p (int): Polynomial order. Typically 2 or 3; must be less than `w`.

    References:
        - Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. 
          *Analytical Chemistry*, 36(8), 1627–1639.
    """
    def __init__(self, w=5, p=2):
        self.w = w
        self.p = p

    def process(self, data):
        """
        Apply the Savitzky-Golay smoothing filter across the time axis of 2D FSCV data.

        This function smooths each voltage sweep (row) independently using the specified window and polynomial degree.
        It is typically applied to enhance signal quality in time-series voltammetry data.

        Args:
            data (np.ndarray): A 2D array representing FSCV scans, where each row corresponds to a voltage sweep 
                               and each column to a time point.

        Returns:
            np.ndarray: A 2D array of the same shape with each row smoothed by the Savitzky-Golay filter.

        Raises:
            ValueError: If the window size is not odd or polynomial degree is invalid.
        """
        data = signal.savgol_filter(data, window_length=self.w, polyorder=self.p, mode='mirror', axis=0)
        return data