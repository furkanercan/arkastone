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
        self.len_kr              = self.len_k + self.len_r
        
        self.qtz_enable          = config["polar"]["quantize"]["enable"]
        self.qtz_chn_max         = config["polar"]["quantize"]["chnl_upper"]
        self.qtz_chn_min         = config["polar"]["quantize"]["chnl_lower"]
        self.qtz_int_max         = config["polar"]["quantize"]["intl_max"]
        self.qtz_int_min         = config["polar"]["quantize"]["intl_min"]

        # These are related to decoding, they should move to decoder part.
        self.max_flips           = config["polar"]["decoder"]["flip_max_iters"]
        self.fast_enable         = config["polar"]["fast_enable"]
        self.nodesize_rate0      = config["polar"]["fast_max_size"]["rate0"]
        self.nodesize_rate1      = config["polar"]["fast_max_size"]["rate1"]
        self.nodesize_rep        = config["polar"]["fast_max_size"]["rep"]
        self.nodesize_spc        = config["polar"]["fast_max_size"]["spc"]
        self.nodesize_ml_0011    = config["polar"]["fast_max_size"]["ml_0011"]
        self.nodesize_ml_0101    = config["polar"]["fast_max_size"]["ml_0101"]

        self.frozen_bits, self.info_indices, self.crc_indices = self.create_polar_indices()


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
            crc_indices (np.ndarray): Updated CRC bit indices (if enabled).
        """

        frozen_bits = np.ones(self.len_n, dtype=int)
        info_indices = self.reliability_indices[:self.len_k]

        if self.en_crc:
            crc_indices = self.reliability_indices[self.len_k:self.len_k + self.len_r]
        else:
            crc_indices = np.array([], dtype=int)

        # Set frozen bits to 0 for info_indices and crc_indices
        frozen_bits[info_indices] = 0
        frozen_bits[crc_indices] = 0

        return frozen_bits, info_indices, crc_indices


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