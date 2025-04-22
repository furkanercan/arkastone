from dataclasses import dataclass, field
import numpy as np

@dataclass
class UncodedConfig:
    len_k: int
    len_n: int = None  # Optional, default to len_k

    def __post_init__(self):
        if self.len_n is None:
            self.len_n = self.len_k

    def __repr__(self):
        return (
            f"UncodedConfig(len_k={self.len_k}, len_n={self.len_n}"
        )
