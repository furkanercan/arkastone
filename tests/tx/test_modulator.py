import numpy as np
from src.tx.core.modulator import Modulator
from src.configs.config_modulation import ModConfig
from src.utils.validation.config_validator import validate_config_modulator

def test_modulator_bpsk():
    #Initialize modulation
    mod_config = ModConfig(type='bpsk', demod_type='soft')
    input_data = np.array([1,  0,  0,  1,  1,  1,  0,  1])
    output_data = np.empty_like(input_data)
    soln_data = np.array([-1, 1,  1,  -1, -1, -1, 1, -1])

    modulator = Modulator(mod_config)
    modulator.modulate(output_data, input_data)
    np.testing.assert_array_equal(output_data, soln_data, err_msg="BPSK modulator test failed.")



def test_modulator_qpsk():

    normalization_factor = 1/np.sqrt(2)

    #Initialize modulation
    mod_config = ModConfig(type='Qpsk', demod_type='soft')
    input_data = np.array([1,  0,  0,  1,  1,  1,  0,  0])
    output_data = np.empty(4, dtype=complex)
    soln_data = np.array([ -1+1j, 1-1j, -1-1j, 1+1j ])*normalization_factor

    modulator = Modulator(mod_config)
    modulator.modulate(output_data, input_data)
    np.testing.assert_array_equal(output_data, soln_data, err_msg="QPSK modulator test failed.")


def test_modulator_16qam():

    normalization_factor = 1/np.sqrt(10)

    #Initialize modulation
    mod_config = ModConfig(type='16QaM', demod_type='sOFt')
    input_data = np.array([1,  0,  0,  1,  1,  1,  0,  0])
    output_data = np.empty(2, dtype=complex)
    soln_data = np.array([ 3-1j, 1-3j ])*normalization_factor

    modulator = Modulator(mod_config)
    modulator.modulate(output_data, input_data)
    np.testing.assert_array_equal(output_data, soln_data, err_msg="16QAM modulator test failed.")


# test_modulator_qpsk()
# test_modulator_16qam()