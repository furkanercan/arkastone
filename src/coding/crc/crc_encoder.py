import numpy as np
from typing import List
from numpy.typing import NDArray
from src.configs.config_crc import CRCConfig

class CRCEncoder:
    def __init__(self, config: CRCConfig):
        """
        Initialize the CRC encoder with the given configuration.
        Args:
            config (CRCConfig): Configuration object containing CRC parameters.
        """
        # self.config = config
        # self.mode = config.mode

        self.crc_length   = config.length
        self.preload_val  = config.preload_val
        self.crc_poly     = config.crc_poly
        self.crc_poly_bin = config.crc_poly_bin

    def encode(self, vec_info: NDArray[np.int_]) -> NDArray[np.int_]:
        """
        Encode input bits with CRC and return only the CRC bits.

        Args:
            vec_info (NDArray[np.int_]): Input binary vector (0s and 1s)

        Returns:
            NDArray[np.int_]: Computed CRC bits
        """
        vec_info = np.asarray(vec_info, dtype=np.int_)  # ensure it's an np array
        len_k = vec_info.size

        # Initialize buffer with input bits and preload value
        vec_info_crc = np.zeros(len_k + self.crc_length, dtype=np.int_)
        vec_info_crc[:len_k] = vec_info
        vec_info_crc[len_k:] = self.preload_val  # preload CRC tail

        # Perform division (XOR with polynomial)
        for i in range(len_k):
            if vec_info_crc[i]:
                vec_info_crc[i:i + len(self.crc_poly_bin)] ^= self.crc_poly_bin

        return vec_info_crc[len_k:]

    def encode_and_append(self, vec_info: NDArray[np.int_]) -> NDArray[np.int_]:
        """
        Append CRC bits to the input vector.

        Args:
            vec_info (NDArray[np.int_]): Input information bits.

        Returns:
            NDArray[np.int_]: Input bits with CRC bits appended.
        """
        vec_info = np.asarray(vec_info, dtype=np.int_)  # make sure it's a NumPy array
        crc = self.encode(vec_info)
        return np.concatenate((vec_info, crc))
