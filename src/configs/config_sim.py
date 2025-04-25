# src/configs/sim_config.py
import numpy as np
from dataclasses import dataclass, field

@dataclass
class SimSweepVals:
    start: float
    end: float
    step: float

    simpoints: np.ndarray = field(init=False)
    len_points: int = field(init=False)

    def __post_init__(self):
        start = self.start
        end = self.end
        step = self.step

        self.simpoints = np.arange(start, end + step, step, dtype=float)
        self.len_points = len(self.simpoints)

@dataclass
class SimLoopConfig:
    num_frames: int
    num_errors: int
    max_frames: int

@dataclass
class SimSaveConfig:
    plot_enable: bool = False
    lutsim_enable: bool = False
    save_output: bool = False

@dataclass
class SimConfig:
    mode: str                   # "dev", "rel"
    sweep_type: str             # "SNR", etc.
    sweep_vals: SimSweepVals
    loop: SimLoopConfig
    save: SimSaveConfig
