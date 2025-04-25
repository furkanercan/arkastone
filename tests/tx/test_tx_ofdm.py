import pytest
import numpy as np
from src.tx.core.tx_ofdm import OFDMTransmitter
from src.common.ofdm import OFDM
from src.configs.config_ofdm import OFDMConfig

config = {
    "num_subcarriers": 16,
    "cyclic_prefix_length": 4
}

def test_transmitter_ofdm():
    len_n = 64

    time_domain_total_length = np.ceil(len_n/config["num_subcarriers"])*(config["num_subcarriers"]+config["cyclic_prefix_length"])
    # Create an OFDM instance and transmitter
    ofdm_config = OFDMConfig(**config)
    ofdm = OFDM(ofdm_config)
    transmitter = OFDMTransmitter(ofdm)
    
    # Generate random modulated data (64 symbols for BPSK)
    modulated_data = 1 - 2 * np.random.randint(0, 2, 64)
    
    # Transmit the data
    transmitted_signal = transmitter.transmit(modulated_data)
    
    # Assert the transmitted signal has the expected shape (time-domain + CP)
    assert transmitted_signal.shape == (time_domain_total_length,)  # 16 subcarriers + 4 CP per transmission
