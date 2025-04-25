import numpy as np
import math
import string
import time

from src.utils.validation.config_loader import ConfigLoader
from src.utils.output_handler import *
from src.utils.create_run_id import *
from src.utils.timekeeper import *

from src.configs.config_modulation import ModConfig
from src.tx.core.modulator import Modulator
from src.rx.core.demodulator import Demodulator

def main_test_mod_demod(config_file):
    """
    This tests the BPSK modulation, without any channel-induced errors.
    An exact match is expected as a result.
    """

    seed = 42
    np.random.seed(seed)
    configloader = ConfigLoader(config_file)
    config = configloader.get()

    mod_config     = ModConfig(**config["mod"])
    modulator   = Modulator(mod_config)
    demodulator = Demodulator(mod_config)

    len_n = 120000 #frame length, make it divisible by 2, 4, 6, 8 for simplicity.

    input_data        = np.empty(len_n, dtype=np.int32) 

    modulated_data_bpsk  = np.empty(len_n, dtype=int)
    modulated_data_qpsk  = np.empty(len_n // 2, dtype=complex) 
    modulated_data_16qam = np.empty(len_n // 4, dtype=complex) 
    modulated_data_64qam = np.empty(len_n // 6, dtype=complex) 
    
    soft_data_bpsk       = np.empty(len_n, dtype=np.float32) 
    soft_data_qpsk       = np.empty(len_n, dtype=np.float32) 
    soft_data_16qam      = np.empty(len_n, dtype=np.float32) 
    soft_data_64qam      = np.empty(len_n, dtype=np.float32) 

    soft_output_bpsk     = np.empty(len_n, dtype=np.int32) 
    soft_output_qpsk     = np.empty(len_n, dtype=np.int32) 
    soft_output_16qam    = np.empty(len_n, dtype=np.int32) 
    soft_output_64qam    = np.empty(len_n, dtype=np.int32) 
       
    hard_output_bpsk     = np.empty(len_n, dtype=np.int32) 
    hard_output_qpsk     = np.empty(len_n, dtype=np.int32) 
    hard_output_16qam    = np.empty(len_n, dtype=np.int32) 
    hard_output_64qam    = np.empty(len_n, dtype=np.int32) 

    var = 0.001

    for _ in range(3):
        input_data[:] = np.random.randint(0, 2, size=len_n)
        modulator.mod_bpsk(modulated_data_bpsk, input_data)
        modulator.mod_qpsk(modulated_data_qpsk, input_data)
        modulator.mod_mqam(modulated_data_16qam, input_data, m=16)

        demodulator.softDemod_bpsk(soft_data_bpsk, modulated_data_bpsk, var)
        demodulator.softDemod_qpsk(soft_data_qpsk, modulated_data_qpsk, var)
        demodulator.softDemod_mqam(soft_data_16qam, modulated_data_16qam, var, m=16)

        soft_output_bpsk[:]  = np.where(soft_data_bpsk < 0, 1, 0)
        soft_output_qpsk[:]  = np.where(soft_data_qpsk < 0, 1, 0)
        soft_output_16qam[:] = np.where(soft_data_16qam < 0, 1, 0)

        demodulator.hardDemod_bpsk(hard_output_bpsk, modulated_data_bpsk)
        demodulator.hardDemod_qpsk(hard_output_qpsk, modulated_data_qpsk)
        demodulator.hardDemod_mqam(hard_output_16qam, modulated_data_16qam, m=16)
        
        # Assert checks for hard outputs (exact match)
        assert np.array_equal(hard_output_bpsk, input_data), f"BPSK hard output does not match input data"
        assert np.array_equal(hard_output_qpsk, input_data), f"QPSK hard output does not match input data"
        assert np.array_equal(hard_output_16qam, input_data), f"16QAM hard output does not match input data"

        assert np.array_equal(soft_output_bpsk, input_data), f"BPSK hard output does not match input data"
        assert np.array_equal(soft_output_qpsk, input_data), f"QPSK hard output does not match input data"
        assert np.array_equal(soft_output_16qam, input_data), f"16QAM hard output does not match input data"
    

def test_mod_demod():
    config_file1 = "tests/endtoend/modulation/config_test_mod_demod.json5"
    main_test_mod_demod(config_file1)


# #Uncomment the following for individual testing or debugging. 
# Comment when pushing or running pytest!
# test_mod_demod() 