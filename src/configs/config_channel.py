# src/configs/channel_config.py

from dataclasses import dataclass

@dataclass
class ChannelConfig:
    type: str       # e.g., "AWGN", "Rayleigh", etc.
    seed: int = 0   # default = 0 (non-deterministic)

    def __post_init__(self):
        self.type = self.type.upper()
        # allowed = {"AWGN", "RAYLEIGH", "RICIAN"}
        allowed = {"AWGN"}
        if self.type not in allowed:
            raise ValueError(f"Unsupported channel type: {self.type}")
