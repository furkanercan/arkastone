import numpy as np
from typing import List
from numpy.typing import NDArray
from src.configs.config_crc import CRCConfig

class CRCDecoder:
    def __init__(self, config: CRCConfig):
        """
        Initialize the CRC decoder with the given configuration.
        Args:
            config (CRCConfig): Configuration object containing CRC parameters.
        """
        self.crc_length   = config.length
        self.preload_val  = config.preload_val
        self.crc_poly     = config.crc_poly
        self.crc_poly_bin = config.crc_poly_bin

    def crc_decode(self, vec_input: NDArray[np.int_], len_k: int) -> bool:
        vec_dec_crc = vec_input.copy()
        for i in range(len_k):
            if vec_dec_crc[i] != 0:
                for j in range(len(self.crc_poly_bin)):
                    vec_dec_crc[i + j] ^= self.crc_poly_bin[j]
        
        crc_pass = all(x == 0 for x in vec_dec_crc)
        return crc_pass