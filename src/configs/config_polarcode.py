import math
import numpy as np
from typing import List
from dataclasses import dataclass, field
from src.utils.validation.import_polarcode_file import import_polarcode_file
from src.configs.config_polarcode_decoder import PolarDecoderConfig
from src.configs.config_polarcode_fast import PolarFastConfig
from src.configs.config_quantization import QuantizeConfig
from src.configs.config_crc import CRCConfig

@dataclass
class PolarCodeConfig:
    len_k     : int 
    polar_file: str
    crc       : CRCConfig
    decoder   : PolarDecoderConfig
    quantize  : QuantizeConfig
    fast_mode : PolarFastConfig
    
    # Derived fields
    len_n       : int        = field(init=False)
    len_logn    : int        = field(init=False)
    len_r       : int        = field(init=False)
    len_kr      : int        = field(init=False)
    rel_idx     : np.ndarray = field(init=False)
    frozen_bits : np.ndarray = field(init=False)
    info_indices: np.ndarray = field(init=False)
    # crc_indices : np.ndarray = field(init=False)

    def __post_init__(self):
        self.rel_idx = import_polarcode_file(self.polar_file)
        self.len_n = len(self.rel_idx)
        self.len_logn = int(math.log2(self.len_n))
        self.len_r = self.crc.length
        self.len_kr = self.len_k + self.len_r
        self.frozen_bits, self.info_indices = self.create_polar_indices()


    def create_polar_indices(self):
        """
        Create frozen and information bit indices for the polar code.
        """

        frozen_bits = np.ones(self.len_n, dtype=int)
        # info_indices = np.array(self.rel_idx[:self.len_k])

        if self.crc.enable:
            # crc_indices = np.array(self.rel_idx[self.len_k:self.len_k + self.crc.length])
            info_indices = np.array(self.rel_idx[:self.len_kr])
        else:
            # crc_indices = np.array([], dtype=int)
            info_indices = np.array(self.rel_idx[:self.len_k])

        frozen_bits[info_indices] = 0
        # frozen_bits[crc_indices] = 0

        return frozen_bits, info_indices
    
    def __repr__(self):
        truncated_info = self.info_indices[:10]
        truncated_frozen = self.frozen_bits[:10]
        truncated_rel = self.rel_idx[:10]
        return (
            f"PolarCode("
            f"info_indices={truncated_info}... (truncated), "
            f"frozen_bits={truncated_frozen}... (truncated))"
            f"reliability indices={truncated_rel}... (truncated))"
        )