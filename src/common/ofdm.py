import numpy as np

class OFDM:
    def __init__(self, config):
        self.num_subcarriers = config["num_subcarriers"]
        self.cyclic_prefix_length = config["cyclic_prefix_length"]

    def add_cyclic_prefix(self, ofdm_symbol):
        """Add a cyclic prefix to the OFDM symbol."""
        cp = ofdm_symbol[-self.cyclic_prefix_length:]
        return np.concatenate([cp, ofdm_symbol])

    def remove_cyclic_prefix(self, received_signal):
        """Remove the cyclic prefix from the received signal."""
        return received_signal[self.cyclic_prefix_length:]

    def perform_ifft(self, frequency_domain_signal):
        """Convert frequency-domain signal to time-domain using IFFT."""
        return np.fft.ifft(frequency_domain_signal, n=self.num_subcarriers)

    def perform_fft(self, time_domain_signal):
        """Convert time-domain signal to frequency-domain using FFT."""
        return np.fft.fft(time_domain_signal, n=self.num_subcarriers)
