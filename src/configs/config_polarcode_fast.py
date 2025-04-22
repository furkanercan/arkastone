from dataclasses import dataclass

@dataclass
class PolarFastConfig:
    enable: bool = False
    max_rate0: int = 1024
    max_rate1: int = 1024
    max_rep: int = 1024
    max_spc: int = 1024
    max_ml_0011: int = 0
    max_ml_0101: int = 0