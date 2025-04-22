from dataclasses import dataclass, field
from typing import Tuple
from typing import List, Optional

@dataclass
class CRCConfig:
    enable: bool              # CRC enable flag
    name: str                 # name your crc, e.g. 'CRC24A'
    length: int               # Number of CRC bits (e.g. 24)
    preload_val: int = 0      # Used in DCI, etc.
    mode: str = 'generic'     # Mode can be '5g' or 'generic'

    # Derived attributes, set in __post_init__
    crc_poly: int = field(init=False)
    crc_poly_bin: List[int] = field(init=False)

    def __post_init__(self):
        if self.mode == '5g':
            self.crc_poly, self.crc_poly_bin = self._instantiate_crcs_5g(self.length)
        elif self.mode == 'generic':
            self.crc_poly, self.crc_poly_bin = self._instantiate_crcs_generic(self.length)
        else:
            raise ValueError(f"Unsupported CRC mode: {self.mode}. Valid: ['5g', 'generic']")

    def _instantiate_crcs_5g(self, len_r: int):
        crc_polys = {
            6:  0x21,
            11: 0x621,
            24: 0xB2B117,
        }
        if len_r not in crc_polys:
            raise ValueError(f"Unsupported 5G CRC length: {len_r}. Valid: {list(crc_polys)}")
        return self._poly_to_bin(crc_polys[len_r], len_r)

    def _instantiate_crcs_generic(self, len_r: int):
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
        if len_r not in crc_polys:
            raise ValueError(f"Unsupported generic CRC length: {len_r}.")
        return self._poly_to_bin(crc_polys[len_r], len_r)

    def _poly_to_bin(self, poly: int, len_r: int):
        CRC_bin = [0] * (len_r + 1)
        CRC_bin[0] = 1
        for i in range(len_r):
            CRC_bin[len_r - i] = (poly >> i) & 1
        return poly, CRC_bin
