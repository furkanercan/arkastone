from dataclasses import dataclass

@dataclass
class PolarDecoderConfig:
    algorithm: str = "SC"
    flip_max_iters: int = 10
    list_size: int = 8