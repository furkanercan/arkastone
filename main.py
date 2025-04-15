from src.utils.validation.config_loader import ConfigLoader
from src.sim.sim_runner import run_simulation_from_config

config_file = "config.json5"
config = ConfigLoader(config_file).get()
run_simulation_from_config(config)
