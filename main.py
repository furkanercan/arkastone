from src.utils.validation.config_loader import ConfigLoader
from src.sim.sim_runner import run_simulation_from_config

intermediate_updates = []
def collect_updates(update):
    intermediate_updates.append(update)

config_file = "config.json5"
config = ConfigLoader(config_file).get()
run_simulation_from_config(config, progress_callback=collect_updates)

print("Final results:", intermediate_updates[-1])