import numpy as np
from src.rx.core.demodulator import Demodulator
from src.configs.config_modulation import ModConfig

def test_demodulator_bpsk_hard():
    config = ModConfig(type='bpsk', demod_type='hard')
    demodulator = Demodulator(config)
    input_data = np.array([-4, 0, 3, 8, 1, -9.1, 0.5, 0.001])
    demod_data = np.empty_like(input_data)
    soln_data = np.array([1, 0, 0, 0, 0, 1, 0, 0])
    demodulator.hardDemod_bpsk(demod_data, input_data)
    
    np.testing.assert_array_equal(demod_data, soln_data, err_msg="BPSK hard demodulator failed")

def test_demodulator_bpsk_soft():
    config = ModConfig(type='bpsk', demod_type='soft')
    demodulator = Demodulator(config)
    input_data = np.array([-4, 0, 3, 8, 1, -9.1, 0.5, 0.001])
    demod_data = np.empty_like(input_data)
    variance = 0.25
    soln_data = 2*input_data/variance
    demodulator.softDemod_bpsk(demod_data, input_data, variance)
    
    np.testing.assert_array_equal(demod_data, soln_data, err_msg="BPSK soft demodulator failed")
