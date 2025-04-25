import pytest
import numpy as np
from src.coding.crc.crc_encoder import CRCEncoder
from src.configs.config_crc import CRCConfig
enable = 1, 
# Source for obtaining CRC polynomials and test vectors:
# https://www.ghsi.de/pages/subpages/Online%20CRC%20Calculation/

def test_crc_encoder_5g_mode():
    config = CRCConfig(enable = 1, name = 'CRC24A', length=24, preload_val=0, mode='5g')
    encoder = CRCEncoder(config)
    info_bits = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    crc_answer = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0]
    crc = encoder.encode(info_bits)
    assert len(crc) == 24
    assert np.array_equal(crc, crc_answer)

def test_crc_encoder_generic_mode():
    config = CRCConfig(enable = 1, name = 'testCRC', length=16, preload_val=0, mode='generic')
    encoder = CRCEncoder(config)
    info_bits = [1, 0, 1, 0, 1, 1, 0, 1]
    crc_answer = [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
    crc = encoder.encode(info_bits)
    print("encoder.crc_poly_bin: ", encoder.crc_poly_bin)
    print("crc: ", crc)
    assert len(crc) == 16
    assert np.array_equal(crc, crc_answer)

def test_crc_encoder_encode_and_append():
    config = CRCConfig(enable = 1, name = 'CRC24A', length=24, preload_val=0, mode='5g')
    encoder = CRCEncoder(config)
    info_bits = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    info_crc_answer = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0]
    appended = encoder.encode_and_append(info_bits)
    assert len(appended) == len(info_crc_answer)
    assert np.array_equal(appended, info_crc_answer)

def test_crc_encoder_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported CRC mode: invalid."):
        config = CRCConfig(enable = 1, name = 'invalid_crc', length=24, preload_val=0, mode='invalid')
        CRCEncoder(config)

def test_crc_encoder_invalid_length():
    with pytest.raises(ValueError, match="Unsupported 5G CRC length: 10."):
        config = CRCConfig(enable = 1, name = 'CRC11', length=10, preload_val=0, mode='5g')
        CRCEncoder(config)


# TODO: Add more tests for different CRC lengths and modes, especially for preload=1 cases.