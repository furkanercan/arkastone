# src/configs/mod_config.py

from dataclasses import dataclass

@dataclass
class ModConfig:
    type: str           # "QPSK", "BPSK", etc.
    demod_type: str     # "soft" or "hard"

    def __post_init__(self):
        allowed_types = {"BPSK", "QPSK", "16QAM"}
        allowed_demod = {"soft", "hard"}

        if self.type.upper() not in allowed_types:
            raise ValueError(f"Unsupported modulation type: {self.type}")
        if self.demod_type.lower() not in allowed_demod:
            raise ValueError(f"Unsupported demod_type: {self.demod_type}")
