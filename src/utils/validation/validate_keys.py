import numpy as np
import warnings
import logging

from src.utils.validation.config_validator_polar import *

logging.basicConfig(
    level=logging.INFO,                  # Set logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Include timestamps
    handlers=[
        logging.FileHandler("run.log"),  # Log file name
        logging.StreamHandler()  # Also log to console
    ]
)

def validate_required_keys(config, required_keys, section_name):
    """
    Validate that all required keys are present in the configuration and have the correct types.

    Args:
        config (dict): The configuration dictionary.
        required_keys (dict): A dictionary of required keys and their expected types.
        section_name (str): The name of the configuration section for error messages.

    Raises:
        ValueError: If a required key is missing.
        TypeError: If a required key has an incorrect type.
    """
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Missing required key in '{section_name}': {key}")
        if not isinstance(config[key], expected_type):
            raise TypeError(f"'{section_name}.{key}' must be of type {expected_type}, got {type(config[key])}")
        
def validate_optional_keys(config, optional_keys, section_name):
    """
    Validate optional keys in the configuration and assign default values if missing.

    Args:
        config (dict): The configuration dictionary.
        optional_keys (dict): A dictionary of optional keys, each mapping to a tuple (expected_type, default_value).
        section_name (str): The name of the configuration section for warnings.

    Returns:
        dict: The updated configuration with defaults assigned for missing optional keys.
    """
    for key, (expected_type, default_value) in optional_keys.items():
        if key not in config:
            # warnings.warn(f"'{section_name}.{key}' not found in configuration. Using default value: {default_value}.", UserWarning)
            # logging.info(
            #     f"'{section_name}.{key}' not found. Defaulting to: {default_value}."
            # )
            config[key] = default_value 
        elif not isinstance(config[key], expected_type):
            raise TypeError(
                f"'{section_name}.{key}' must be of type {expected_type}, got {type(config[key])}"
            )
    return config


