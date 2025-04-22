from dataclasses import dataclass
from src.configs.config_crc import CRCConfig

@dataclass
class PUCCHConfig:
    """
    Configuration for PUCCH (Physical Uplink Control Channel) in NR5G.
    """
    A: int 
    K: int 
    N: int 
    E: int 
    G: int 
    rm: str 
    seg: bool 
    crc_config: CRCConfig 
    pc_indices: list  
    info_indices: list 
    frozen_indices: list 
    channel_interleaved_indices: list 
    Gmat_kxN: list 
