import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.validation.config_loader import ConfigLoader
from src.sim.sim_runner import run_simulation_from_config

config_file = "configs/config_polar.json5"
config = ConfigLoader(config_file).get()
run_simulation_from_config(config)
