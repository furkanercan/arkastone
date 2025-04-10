from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CRCConfig:
    name: str                 # e.g. 'CRC24A'
    length: int               # Number of CRC bits (e.g. 24)
    preload_val: int = 0      # Used in DCI, etc.
    mode: str = '5g'          # Mode can be '5g' or 'generic'