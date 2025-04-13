import numpy as np
import math

class PolarCode:
    def __init__(self, config):
        """
        Initialize the PolarCode with information indices and frozen bits.
        """
        self.len_k = config["len_k"]

        self.reliability_indices = config["polar"]["rel_idx"]
        self.len_n               = config["polar"]["len_n"]
        self.len_logn            = config["polar"]["len_logn"]
        self.en_crc              = config["polar"]["crc"]["enable"]
        self.len_r               = config["polar"]["crc"]["length"]
        
        self.qtz_enable          = config["polar"]["quantize"]["enable"]
        self.qtz_chn_max         = config["polar"]["quantize"]["chnl_upper"]
        self.qtz_chn_min         = config["polar"]["quantize"]["chnl_lower"]
        self.qtz_int_max         = config["polar"]["quantize"]["intl_max"]
        self.qtz_int_min         = config["polar"]["quantize"]["intl_min"]

        self.fast_enable         = config["polar"]["fast_enable"]
        self.nodesize_rate0      = config["polar"]["fast_max_size"]["rate0"]
        self.nodesize_rate1      = config["polar"]["fast_max_size"]["rate1"]
        self.nodesize_rep        = config["polar"]["fast_max_size"]["rep"]
        self.nodesize_spc        = config["polar"]["fast_max_size"]["spc"]
        self.nodesize_ml_0011    = config["polar"]["fast_max_size"]["ml_0011"]
        self.nodesize_ml_0101    = config["polar"]["fast_max_size"]["ml_0101"]

        self.frozen_bits, self.info_indices = self.create_polar_indices()


    def create_polar_indices(self):
        """
        Create frozen and information bit indices for the polar code.

        Args:
            len_n (int): Block length (N).
            len_k (int): Number of information bits (K).
            en_crc (bool): Whether CRC is enabled.
            len_r (int): Length of CRC (if enabled).

        Raises:
            ValueError: If any of the inputs are invalid.

        Returns:
            frozen_bits (np.ndarray): Updated frozen bit vector.
            info_indices (np.ndarray): Updated information bit indices.
        """

        frozen_bits = np.ones(self.len_n, dtype=int)
        vec_polar_info = np.ones(self.len_n, dtype=int)

        len_i = self.len_k
        if(self.en_crc):
            len_i += self.len_r
        for num, index in enumerate(self.reliability_indices[:len_i], start=0):
            frozen_bits[index] = 0
        for num, index in enumerate(self.reliability_indices[len_i:], start=len_i):
            vec_polar_info[index] = 0 
            
        info_indices = np.nonzero(vec_polar_info)[0]

        return frozen_bits, info_indices


    def __repr__(self):
        truncated_info = self.info_indices[:10]
        truncated_frozen = self.frozen_bits[:10]
        truncated_rel = self.reliability_indices[:10]
        return (
            f"PolarCode("
            f"info_indices={truncated_info}... (truncated), "
            f"frozen_bits={truncated_frozen}... (truncated))"
            f"reliability indices={truncated_rel}... (truncated))"
        )