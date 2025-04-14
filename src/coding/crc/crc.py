import numpy as np

def hex_to_bin_list(hex_str):
    bin_str = bin(int(hex_str, 16))[2:]  # Convert hex to binary string
    return [int(bit) for bit in bin_str]  # Convert binary string to list of integers

def crc_encode(vec_info, vec_info_crc, CRC_bin, len_k, preload_val=0):
    vec_info_crc[:len_k] = vec_info
    vec_info_crc[len_k:] = 0

    if(preload_val == 1): #preload_val can be either 0 or 1
        vec_info_crc[len_k:] = 1

    # for idx, val in enumerate(vec_info_crc):
    #     print(val, "/")
    # print("\n")

    for i in range(len_k):
        if vec_info_crc[i] != 0:
            for j in range(len(CRC_bin)):
                vec_info_crc[i + j] ^= CRC_bin[j]
    
    vec_info_crc[:len_k] = vec_info

    return vec_info_crc

def compute_crc_5g_polar(vec_info: list[int], len_r=24, prefill_val=0) -> list[int]:
    """
    Computes a CRC of length len_r using 5G-compliant polynomial and optional preload.

    Args:
        vec_info (list[int]): Information bits (0 or 1)
        len_r (int): CRC length, default is 24 (used in polar coding)
        prefill_val (int): Preload value for CRC register (e.g., 0xFFFFFF for DCI)

    Returns:
        list[int]: The CRC bits as a list of 0s and 1s
    """
    _, crc_poly = instantiate_crcs_5g(len_r)
    len_k = len(vec_info)
    vec_info_crc = np.zeros(len_k + 24, dtype=int)
    crc_encoded = crc_encode(vec_info, vec_info_crc, crc_poly, len_k, prefill_val)
    return crc_encoded[len_k:].tolist()


def instantiate_crcs_5g(len_r):
    CRC_bin = [0] * (len_r + 1)
    
    crc_polys = {
        6:  0x21,  # 5G
        11: 0x621,  # 5G
        # 16: 0x1021,  # 5G
        24: 0xB2B117,  # 5G
    }

    if len_r not in crc_polys:
        raise ValueError(f"Unsupported 5G CRC length: {len_r}. Valid options are {list(crc_polys.keys())}.")
    
    CRC_poly = crc_polys.get(len_r, 0)
    
    CRC_bin[0] = 1
    for i in range(len_r):
        CRC_bin[len_r - i] = (CRC_poly >> i) & 1

    return CRC_poly, CRC_bin


def instantiate_crcs(len_r):
    CRC_bin = [0] * (len_r + 1)
    
    crc_polys = {
        0:  0x0,
        1:  0x1,
        2:  0x3,
        3:  0x3,
        4:  0x3,
        5:  0x15,
        6:  0x21,  # 5G
        7:  0x09,
        8:  0xD5,
        9:  0x119,
        10: 0x233,
        11: 0x621,  # 5G
        12: 0x80F,
        13: 0x1CF5,
        14: 0x202D,
        15: 0x4599,
        16: 0x1021,  # 5G
        17: 0x1685B,
        18: 0x23979,
        19: 0x6FB57,
        20: 0xB5827,
        21: 0x102899,
        22: 0x308FD3,
        23: 0x540DF0,
        24: 0xB2B117,  # 5G
        25: 0x101690C,
        26: 0x33C19EF,
        27: 0x5E04635,
        28: 0x91DC1E3,
        29: 0x16DFBF51,
        30: 0x2030B9C7,
        31: 0x6C740B8D,
        32: 0x04C11DB7,
        40: 0x0004820009,
        64: 0x000000000000001B
    }
    
    CRC_poly = crc_polys.get(len_r, 0)
    
    CRC_bin[0] = 1
    for i in range(len_r):
        CRC_bin[len_r - i] = (CRC_poly >> i) & 1

    return CRC_poly, CRC_bin