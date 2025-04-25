import json
from src.utils.validation.validation_manager import validate_config

# Loads config, Validates config, Creates additional config

class ConfigLoader:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.raw_config = json.load(f) #self, because tests are using this.


        self.config = validate_config(self.raw_config)

    def get(self):
        return self.config
