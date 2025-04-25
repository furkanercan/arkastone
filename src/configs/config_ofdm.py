from dataclasses import dataclass

@dataclass
class OFDMConfig:
    num_subcarriers: int
    cyclic_prefix_length: int

    def __post_init__(self):
        if self.num_subcarriers <= 0:
            raise ValueError("num_subcarriers must be > 0")
        if self.cyclic_prefix_length < 0:
            raise ValueError("cyclic_prefix_length must be >= 0")
