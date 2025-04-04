from src.coding.crc import *

def test_crc24_polar_instantiation():
    poly, crc_bin = instantiate_crcs(24)
    assert poly == 0xB2B117
    assert len(crc_bin) == 25
    assert crc_bin == [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
    assert crc_bin[0] == 1  # Always leading 1
    # Check specific bits from known polynomial
    assert crc_bin[-1] == 1  # x^0 term
    assert crc_bin[-2] == 1  # x^1
    assert crc_bin[-3] == 1  # x^1
    assert crc_bin[-4] == 0  # x^3 

    assert crc_bin[3] == 1  # x^3 
    assert crc_bin[2] == 0  # x^3 
    assert crc_bin[1] == 1  # x^3 
    assert crc_bin[0] == 1  # x^3 

def test_crc_encode_manual():
    info_bits = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    crc_answer = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0]
    
    len_k = len(info_bits)
    vec_info_crc = np.zeros(len_k + 24, dtype=int)
    _, crc_bin = instantiate_crcs(24)

    crc_encoded = crc_encode(info_bits, vec_info_crc, crc_bin, len_k)
    
    assert (crc_encoded[:len_k] == info_bits).all() # Check if original message is intact 
    crc_bits = crc_encoded[len_k:] 
    # print("crc bits: ", crc_bits)
    assert len(crc_bits) == 24 # Check CRC is 24 bits
    assert all(b in (0, 1) for b in crc_bits) # Check CRC is all binary
    assert crc_bits.tolist() == crc_answer # Check if the result checks out

def test_crc_5g_polar_default_preload():
    # info_bits = [0] * 128
    len_r = 24
    info_bits = hex_to_bin_list('0xABCD0123CDEF4567')
    crc_answer = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    # print(info_bits)
    crc = compute_crc_5g_polar(info_bits, len_r)
    assert len(crc) == 24
    assert all(b in (0, 1) for b in crc)
    assert crc == crc_answer # Check if the result checks out

def test_crc_5g_polar_dci_preload():
    len_r = 24
    info_bits = hex_to_bin_list('0x0123CDEF4567ABCD')
    crc_answer = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    crc = compute_crc_5g_polar(info_bits, len_r, prefill_val=1)
    assert len(crc) == 24
    assert all(b in (0, 1) for b in crc)
    print(crc)
    assert crc == crc_answer # Check if the result checks out
    #TODO: find/create independent CRC calculator with preload function, compare values


# TODO: create a set of tests that goes over 1000s of checks for comprehensiveness.