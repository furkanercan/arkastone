import warnings
import numpy as np
import math
from src.utils.validation.validate_keys import validate_required_keys
from src.utils.validation.validate_keys import validate_optional_keys
from src.utils.validation.import_polarcode_file import import_polarcode_file
from src.coding.crc.crc import instantiate_crcs

def validate_config_polar(config):
    required_keys = {
        "polar_file": str,
    }

    optional_keys = {
        "crc": dict,      # Delegate to `validate_crc_config`
        "decoder": dict,  # Delegate to `validate_decoder_config`
        "quantize": dict,    # Delegate to `validate_quant_config`
        "fast_enable": bool, # Delegate to `fast enable`
        "fast_max_size": dict # Delegate to `fast max size`
    }

    validate_required_keys(config, required_keys, "polar")
    
    config["rel_idx"] = import_polarcode_file(config["polar_file"])
    config["len_n"] = config["rel_idx"][0] + 1
    config["len_logn"] = int(math.log2(config["len_n"]))

    # Validate optional nested sections
    if "crc" in config:
        config["crc"] = validate_config_polar_crc(config["crc"])
    if "decoder" in config:
        config["decoder"] = validate_config_polar_decoder(config["decoder"])
    if "quantize" in config:
        config["quantize"] = validate_config_polar_quantize(config["quantize"])
    if "fast_enable" in config:
        config["fast_enable"] = validate_config_polar_fast_enable(config["fast_enable"])
    if "fast_max_size" in config:
        config["fast_max_size"] = validate_config_polar_fast_max_size(config["fast_max_size"], config["fast_enable"])

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

    # optional_keys = {
    #     "poly_hex": (int, 0xD5)
    # }

    validate_required_keys(config, required_keys, "polar.crc")
    # validate_optional_keys(config, optional_keys, "polar.crc")

    enable = config["enable"]

    if enable < 0:
        raise ValueError(f"'polar.crc.enable' ({enable}) must be a non-negative value.")
    
    config["poly"], config["binary"] = instantiate_crcs(config["length"])

    return config



def validate_config_polar_quantize(config):
    optional_keys = {
        "enable": (bool, False),
        "bits_chnl": (int, 5),
        "bits_intl": (int, 6),
        "bits_frac": (int, 1)
    }

    validate_optional_keys(config, optional_keys, "polar.quantize")
    
    config["step"]         =    2 **  config["bits_frac"]
    config["chnl_upper"]   = (  2 ** (config["bits_chnl"] -1) - 1)/config["step"]
    config["chnl_lower"]   = (-(2 ** (config["bits_chnl"] -1)))//  config["step"]
    config["intl_max"]     = (  2 ** (config["bits_intl"] -1) - 1)/config["step"]
    config["intl_min"]     = (-(2 ** (config["bits_intl"] -1)))//  config["step"]

    return config


def validate_config_polar_fast_enable(config):
    """
    Validates the 'fast_enable' configuration. If the input is a boolean, it is returned directly.
    Otherwise, it validates the dictionary structure.
    """
    if isinstance(config, bool):
        # If config is a boolean, return it directly
        return config

    # # If config is a dictionary, validate its keys
    # required_keys = {
    #     "enable": bool
    # }

    # validate_required_keys(config, required_keys, "polar.fast_enable")

    return config


def validate_config_polar_fast_max_size(config, enabled):

    optional_keys = {
        "rate0": (int, 4),
        "rate1": (int, 4),
        "rep": (int, 4),
        "spc": (int, 4)
    }

    validate_optional_keys(config, optional_keys, "polar.fast_max_size")

    # Further validate only if the corresponding fast_enable key is True
    for key, value in config.items():
        if key in optional_keys:
            if (enabled == True):  # Check if the corresponding fast_enable key is True
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
