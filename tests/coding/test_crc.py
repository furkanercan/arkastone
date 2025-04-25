import numpy as np
from src.tx.nr5g.polar.components.rnti_scrambling import *
from src.configs.config_crc import CRCConfig
from src.coding.crc.crc_encoder import CRCEncoder  

def test_crc24_polar_instantiation():
    crc_config = CRCConfig(
        enable= 1,
        name='crc',
        length=24,
        preload_val=0,  
        mode='generic'
    )
    crc = CRCEncoder(crc_config)

    assert crc.crc_poly == 0xB2B117
    assert len(crc.crc_poly_bin) == 25
    assert crc.crc_poly_bin == [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
    assert crc.crc_poly_bin[0] == 1  # Always leading 1
    # Check specific bits from known polynomial
    assert crc.crc_poly_bin[-1] == 1  # x^0 term
    assert crc.crc_poly_bin[-2] == 1  # x^1
    assert crc.crc_poly_bin[-3] == 1  # x^1
    assert crc.crc_poly_bin[-4] == 0  # x^3 

    assert crc.crc_poly_bin[3] == 1  # x^3 
    assert crc.crc_poly_bin[2] == 0  # x^3 
    assert crc.crc_poly_bin[1] == 1  # x^3 
    assert crc.crc_poly_bin[0] == 1  # x^3 

def test_crc_encode_manual():
    info_bits = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    crc_answer = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0]
    
    len_k = len(info_bits)
    # vec_info_crc = np.zeros(len_k + 24, dtype=int)
    crc_config = CRCConfig(
        enable= 1,
        name='crc',
        length=24,
        preload_val=0,  
        mode='generic'
    )
    crc = CRCEncoder(crc_config)

    crc_encoded = crc.encode_and_append(info_bits)
    
    assert (crc_encoded[:len_k] == info_bits).all() # Check if original message is intact 
    crc_bits = crc_encoded[len_k:] 
    # print("crc bits: ", crc_bits)
    assert len(crc_bits) == 24 # Check CRC is 24 bits
    assert all(b in (0, 1) for b in crc_bits) # Check CRC is all binary
    assert crc_bits.tolist() == crc_answer # Check if the result checks out

def test_crc_5g_polar_default_preload():

    crc_config = CRCConfig(
        enable= 1,
        name='crc',
        length=24,
        preload_val=0,  
        mode='generic'
    )
    crc = CRCEncoder(crc_config)

    # info_bits = [0] * 128

    info_bits = [int(bit) for bit in bin(int('ABCD0123CDEF4567', 16))[2:].zfill(64)]
    crc_answer = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    # print(info_bits)
    crc_encoded = crc.encode(info_bits)
    assert len(crc_encoded) == 24
    assert all(b in (0, 1) for b in crc_encoded)
    assert np.array_equal(crc_encoded, crc_answer)

def test_crc_5g_polar_dci_preload():
    crc_config = CRCConfig(
        enable= 1,
        name='crc',
        length=24,
        preload_val=1,  
        mode='generic'
    )
    crc = CRCEncoder(crc_config)

    info_bits = [int(bit) for bit in bin(int('0123CDEF4567ABCD', 16))[2:].zfill(64)]
    crc_answer = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    crc_encoded = crc.encode(info_bits)
    assert len(crc_encoded) == 24
    assert all(b in (0, 1) for b in crc_encoded)
    print(crc_encoded)
    assert np.array_equal(crc_encoded, crc_answer)
    #TODO: find/create independent CRC calculator with preload function, compare values


# TODO: create a set of tests that goes over 1000s of checks for comprehensiveness.







def test_rnti_scrambling():
    crc = [0] * 24
    rnti = 0xABCD
    scrambled = rnti_scramble_crc(crc, rnti)
    assert scrambled[:8] == [0]*8
    assert scrambled[8:] == [(rnti >> i) & 1 for i in reversed(range(16))]
