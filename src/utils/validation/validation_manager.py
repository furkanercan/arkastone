from src.utils.validation.config_validator import *

VALIDATORS = {
    "code": validate_config_code,           # Validate the 'code' section
    "mod": validate_config_modulator,       # Validate the 'mod' section
    "channel": validate_config_channel,     # Validate the 'channel' section
    "sim": validate_config_sim,              # Validate the 'sim' section
    "ofdm": validate_config_ofdm              # Validate the 'sim' section
}

def validate_config(config):
    """
    Validate the entire configuration by delegating to section-specific validators.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: The validated configuration.

    Raises:
        ValueError: If a required section is missing or a validator raises an error.
    """
    validated_config = {}
    for section, validator in VALIDATORS.items():
        if section not in config:
            raise ValueError(f"Missing required configuration section: '{section}'")
        validated_config[section] = validator(config[section])
    return validated_config