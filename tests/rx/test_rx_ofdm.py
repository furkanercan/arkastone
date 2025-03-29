import pytest
import numpy as np
from src.tx.tx_ofdm import OFDMTransmitter
from src.rx.rx_ofdm import OFDMReceiver
from src.common.ofdm import OFDM

config = {
    "num_subcarriers": 16,
    "cyclic_prefix_length": 4
}

def test_receiver():
    len_n = 64
    # Create an OFDM instance and receiver
    ofdm = OFDM(config)
    receiver = OFDMReceiver(ofdm, len_n)
    
    # Generate random modulated data (len_n symbols for BPSK)
    modulated_data = 1 - 2 * np.random.randint(0, 2, len_n)
    
    # Create transmitter and transmit signal
    transmitter = OFDMTransmitter(ofdm)
    transmitted_signal = transmitter.transmit(modulated_data)
    
    # Simulate receiving the signal (no noise)
    received_signal = transmitted_signal
    
    # Receive and process the signal
    recovered_data = receiver.receive(received_signal)
    
    # Assert that the recovered data has the correct shape (frequency-domain)
    assert recovered_data.shape == (len_n,) 

