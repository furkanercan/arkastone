import pytest
import numpy as np
from src.common.ofdm import OFDM
from src.configs.config_ofdm import OFDMConfig

config = {
    "num_subcarriers": 16,
    "cyclic_prefix_length": 4
}

def test_ifft():
    # Create an OFDM instance
    ofdm_config = OFDMConfig(**config)
    ofdm = OFDM(ofdm_config)

    # Create some frequency-domain data (16 subcarriers)
    frequency_domain_signal = np.random.randn(16) + 1j * np.random.randn(16)
    
    # Perform IFFT
    time_domain_signal = ofdm.perform_ifft(frequency_domain_signal)
    
    # Assert that the time-domain signal is the correct length
    assert time_domain_signal.shape == (16,) 
    
def test_fft():
    # Create an OFDM instance
    ofdm_config = OFDMConfig(**config)
    ofdm = OFDM(ofdm_config)

    # Create some time-domain data (16 samples)
    time_domain_signal = np.random.randn(16) + 1j * np.random.randn(16)
    
    # Perform FFT
    frequency_domain_signal = ofdm.perform_fft(time_domain_signal)
    
    # Assert that the frequency-domain signal is the correct length
    assert frequency_domain_signal.shape == (16,)

def test_add_remove_cyclic_prefix():
    # Create an OFDM instance
    ofdm_config = OFDMConfig(**config)
    ofdm = OFDM(ofdm_config)

    # Create a random time-domain signal (16 samples)
    time_domain_signal = np.random.randn(16) + 1j * np.random.randn(16)
    
    # Add cyclic prefix
    signal_with_cp = ofdm.add_cyclic_prefix(time_domain_signal)
    
    # Assert that the signal length is correct (16 + 4 CP samples)
    assert signal_with_cp.shape == (20,)
    
    # Remove cyclic prefix
    signal_no_cp = ofdm.remove_cyclic_prefix(signal_with_cp)
    
    # Assert that the signal length is back to the original (16 samples)
    assert signal_no_cp.shape == (16,)
