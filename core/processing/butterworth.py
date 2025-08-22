import numpy as np
from .base import Processor
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class ButterworthFilter(Processor):
    def __init__(self, p=4, fs_x=10, fs_y=500000):
        """
        Apply a 2D low-pass Butterworth filter to FSCV color plot data. The implementation is inspired by the paper Novel, User-Friendly Experimental and Analysis Strategies for Fast Voltammetry: 1. The Analysis Kid for FSCV by Mena et al. (2021) 
        https://doi.org/10.1021/acs.analchem.1c01258

        Args:
            p (int): Filter order.
            fs_x (int): Sampling rate in time direction.
            fs_y (int): Sampling rate in voltage direction.

        Methods:
            process(data): Returns filtered data.
            visualize_cutoff(data): Shows cutoff region on FFT spectrum.
        """
        self.p = p

    def process(self, data):
        """
        Apply a 2D Butterworth filter to the input data.

        Args:
            data (np.ndarray): 2D FSCV data array.

        Returns:
            np.ndarray: Filtered 2D data.
        """
        rows, cols = data.shape

        # Amount of padding — you can tweak this
        pad_y = int(0.1 * rows)   # 10% vertical padding
        pad_x = int(0.1 * cols)   # 10% horizontal padding

        # Apply reflect padding and store it
        padded_data = np.pad(data, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')

        # 1. Perform 2D FFT and shift the zero frequency to the center
        F = np.fft.fft2(padded_data)
        F_shifted = np.fft.fftshift(F)

        fs_x = 10  # Acquisition frequency in x direction
        fs_y = 500000 # Update rate in y direction

        padded_rows, padded_cols = padded_data.shape

        # 2. Frequency vectors in Hz
        fx = np.fft.fftfreq(padded_cols, d=1/fs_x)
        fy = np.fft.fftfreq(padded_rows, d=1/fs_y) 

        wx = np.fft.fftshift(fx)
        wy = np.fft.fftshift(fy)

        WX, WY = np.meshgrid(wx, wy)

        # 3. 15% cutoff
        cx = 0.15 * (fs_x / 2)
        cy = 0.15 * (fs_y / 2)

        #print(f"Cutoff frequencies: cx={cx}, cy={cy}") -> cx=0.75, cy=37500.0
        # 4. Compute the custom 2D Butterworth transfer function
        H = 1 / (1 + (WX / cx)**(2) + (WY / cy)**(2))**self.p

        # 5. Apply transfer function in frequency domain
        F_filtered = F_shifted * H

        # 6. Inverse FFT to return to spatial domain
        filtered = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real

        # 7. Crop to original size (remove padding)
        filtered_cropped = filtered[pad_y:-pad_y, pad_x:-pad_x]

        #self.visualize_cutoff(data)

        return filtered_cropped
    
    def visualize_cutoff(self, data):
        """
        This visualization function plots the 2D FFT magnitude spectrum of the input data with an overlaid Butterworth cutoff region.
        It currently does not work properly as it does not warp the ellipse to the correct aspect ratio.
        It is a placeholder for future work.
        """
        rows, cols = data.shape

        fs_x = 10  # Time axis sampling rate
        fs_y = 500_000  # Voltage axis sampling rate

        # 1. FFT and shift
        F = np.fft.fft2(data)
        F_shifted = np.fft.fftshift(F)
        magnitude = 20 * np.log10(np.abs(F_shifted) + 1e-12)  # in dB

        # 2. Frequency axes
        fx = np.fft.fftshift(np.fft.fftfreq(cols, d=1/fs_x))
        fy = np.fft.fftshift(np.fft.fftfreq(rows, d=1/fs_y))
        WX, WY = np.meshgrid(fx, fy)

        # 3. Cutoff frequencies
        cx = 0.15 * (fs_x / 2)
        cy = 0.15 * (fs_y / 2)

        # 4. Plot magnitude spectrum
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            magnitude,
            extent=[fx[0], fx[-1], fy[0], fy[-1]],
            origin='lower',
            cmap='gray',
            aspect='auto'
        )
        ax.set_title('2D FFT Magnitude with Butterworth Cutoff')
        ax.set_xlabel('Frequency (Hz) — Time Axis')
        ax.set_ylabel('Frequency (Hz) — Voltage Axis')
        plt.colorbar(im, ax=ax, label='Magnitude (dB)')

        # 5. Overlay Butterworth cutoff region
        ellipse = Ellipse(
            (0, 0),
            width=2 * cx,
            height=2 * cy,
            edgecolor='red',
            facecolor='none',
            linestyle='--',
            linewidth=2,
            label='Cutoff region'
        )
        ax.add_patch(ellipse)
        ax.legend()

        x_span = fx.max() - fx.min()
        y_span = fy.max() - fy.min()
        ax.set_aspect(x_span / y_span)

        plt.tight_layout()
        plt.show()





