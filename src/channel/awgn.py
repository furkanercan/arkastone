import numpy as np
import math
from src.configs.config_channel import ChannelConfig
from src.configs.config_sim import SimConfig

class ChannelAWGN:
    def __init__(self, config_chn: ChannelConfig, config_sim: SimConfig):
        self.type = config_chn.type.lower()
        self.seed = config_chn.seed
        np.random.seed(self.seed)

        # Borrow the following two parameters from the simulation config
        snr_type = config_sim.sweep_type.lower()
        simpoints = config_sim.sweep_vals.simpoints
        simpoints = np.array(simpoints)
        
        if(snr_type == "snr"):
            snr_linear = 10 ** (simpoints / 10) 
            self.variance = 1/snr_linear
            self.stdev = np.sqrt(self.variance)
        else:
            TypeError("Current version only supports SNR(dB) type simulation!")

        self.mean = 0

    def apply_awgn(self, modulated_data, stdev, variance):
        """
        Add AWGN to the transmitted signal.

        Args:
            modulated_data (np.ndarray): Modulated symbols (real or complex).

        Returns:
            np.ndarray: Noisy received symbols.
        """
        if np.iscomplexobj(modulated_data):
            # Use complex noise for complex modulated symbols (e.g., QPSK)
            noise = self.generate_complex_noise(len(modulated_data), variance)
        else:
            # Use real noise for real modulated symbols (e.g., BPSK)
            noise = np.random.normal(0, stdev, len(modulated_data))
        return modulated_data + noise
    

    def generate_complex_noise(self, size, variance):
        """
        Generate complex AWGN noise.

        Args:
            size (int): Number of noise samples to generate.
            noise_variance (float): Noise variance (sigma^2).

        Returns:
            np.ndarray: Complex noise samples.
        """
        real_noise = np.random.normal(0, np.sqrt(variance / 2), size)
        imag_noise = np.random.normal(0, np.sqrt(variance / 2), size)
        return real_noise + 1j * imag_noise