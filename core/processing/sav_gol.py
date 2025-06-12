from .base import Processor
import scipy.signal as signal

class SavitzkyGolayFilter(Processor):
    def __init__(self, w=5, p=2):
        self.w = w
        self.p = p

    def process(self, data):
        data = signal.savgol_filter(data, window_length=self.w, polyorder=self.p, mode='mirror', axis=0)
        return data