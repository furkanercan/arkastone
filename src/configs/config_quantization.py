from dataclasses import dataclass, field

@dataclass
class QuantizeConfig:
    enable: bool = False
    bits_chnl: int = 5
    bits_intl: int = 6
    bits_frac: int = 1
    
    chnl_upper: float = field(init=False)
    chnl_lower: float = field(init=False)
    intl_max: float = field(init=False)
    intl_min: float = field(init=False)

    def __post_init__(self):
        step = 2 ** self.bits_frac
        self.chnl_upper = (2 ** (self.bits_chnl - 1) - 1) / step
        self.chnl_lower = -(2 ** (self.bits_chnl - 1)) // step
        self.intl_max = (2 ** (self.bits_intl - 1) - 1) / step
        self.intl_min = -(2 ** (self.bits_intl - 1)) // step