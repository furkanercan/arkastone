import warnings
import numpy as np
import math
from src.utils.validation.validate_keys import validate_required_keys
from src.utils.validation.validate_keys import validate_optional_keys
from src.configs.config_crc import CRCConfig
from src.coding.crc.crc_encoder import CRCEncoder 

def validate_config_polar(config):
    required_keys = {
        "polar_file": str,
    }

    # optional_keys = {
    #     "crc": dict,      # Delegate to `validate_crc_config`
    #     "decoder": dict,  # Delegate to `validate_decoder_config`
    #     "quantize": dict, # Delegate to `validate_quant_config`
    #     "fast_mode": dict # Delegate to `validate_config_polar_fast_mode
    # }

    validate_required_keys(config, required_keys, "polar")

    # Validate optional nested sections
    if "crc" in config:
        config["crc"] = validate_config_polar_crc(config["crc"])
    if "decoder" in config:
        config["decoder"] = validate_config_polar_decoder(config["decoder"])
    if "quantize" in config:
        config["quantize"] = validate_config_polar_quantize(config["quantize"])
    if "fast_mode" in config:
        config["fast_mode"] = validate_config_polar_fast_mode(config["fast_mode"])

    return config



def validate_config_polar_decoder(config):
    required_keys = {
        "algorithm": str
    }

    optional_keys = {
        "flip_max_iters": (int, 10),
        "list_size": (int, 8)
    }

    validate_required_keys(config, required_keys, "polar.decoder")
    validate_optional_keys(config, optional_keys, "polar.decoder")

    return config




def validate_config_polar_crc(config):
    required_keys = {
        "enable": bool,
        "length": int
    }

    optional_keys = {
        "name"  : (str, 'default'),
        "mode"  : (str, 'generic'),
        "preload_val"  : (int, 0)
    }

    validate_required_keys(config, required_keys, "polar.crc")
    validate_optional_keys(config, optional_keys, "polar.crc")
    
    # # Note: config bool not recognized by github actions, comment out for now
    # if config["enable"] not in (0, 1):
    #     raise ValueError(f"'polar.crc.enable' ({config["enable"]}) must be a boolean value.")
    # if config["preload_val"] not in (0, 1):
    #     raise ValueError(f"'polar.crc.enable' ({config["preload_val"]}) must be a boolean value.")

    return config



def validate_config_polar_quantize(config):
    optional_keys = {
        "enable": (bool, False),
        "bits_chnl": (int, 5),
        "bits_intl": (int, 6),
        "bits_frac": (int, 1)
    }

    validate_optional_keys(config, optional_keys, "polar.quantize")
    
    return config


def validate_config_polar_fast_mode(config):

    required_keys = {
        "enable": bool,
    }

    optional_keys = {
        "rate0": (int, 4),
        "rate1": (int, 4),
        "rep": (int, 4),
        "spc": (int, 4)
    }

    validate_required_keys(config, required_keys, "polar.fast_mode")
    validate_optional_keys(config, optional_keys, "polar.fast_mode")

    # Further validate only if the corresponding fast_enable key is True
    for key, value in config.items():
        if key in optional_keys:
            if (config["enable"] == True):  # Check if the corresponding fast_enable key is True
                if value != 0:
                    if value < 4:
                        raise ValueError(
                            f"'polar.fast_max_size.{key}' ({value}) must be at least 4."
                        )
                    if not (value & (value - 1)) == 0:
                        raise ValueError(
                            f"'polar.fast_max_size.{key}' ({value}) must be a power of 2."
                        )
            else:
                # Skip validation if fast_enable[key] is False
                config[key] = optional_keys[key][1]  # Assign the default value silently

    return config
