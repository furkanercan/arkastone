import os
from src.utils.validation.config_validator_polar import *
from src.utils.validation.validate_keys import *

def validate_config_code(config_code):
    required_keys = {
        "type": str,
        "len_k": int,
    }

    optional_keys = {
        "polar": dict # delegate to validate_polar
    }

    validate_required_keys(config_code, required_keys, "code")
    
    len_k = config_code["len_k"]
    if len_k < 0:
        raise ValueError(f"'polar.len_k' ({len_k}) must be a non-negative value.")

    if "polar" in config_code:
        config_code["polar"] = validate_config_polar(config_code["polar"])
    # Other codes will follow here. (LDPC, Turbo, CRC, RS, BCH, OSD, etc.)

    return config_code




def validate_config_modulator(config_mod):
    required_keys = {
        "type": str,
        "demod_type": str
    }
    validate_required_keys(config_mod, required_keys, "mod")
    return config_mod




def validate_config_channel(config_chn):
    required_keys = {
        "type": str
    }
    optional_keys = {
        "seed": (int, 42)  # Default value is 42
    }

    validate_required_keys(config_chn, required_keys, "config_chn")
    validate_optional_keys(config_chn, optional_keys, "config_chn")

    return config_chn



def validate_config_sim(config_sim):
    required_keys = {
        "mode": str,
        "sweep_type": str,
        "sweep_vals": dict,
        "loop": dict,   # Delegate to `validate_loop_config`
        "save": dict    # Delegate to `validate_save_config`
    }
    # optional_keys = {
    #     "mode": (str, "dev")  # Default value is dev
    # }


    validate_required_keys(config_sim, required_keys, "sim")
    # validate_optional_keys(config_sim, optional_keys, "sim")

    config_sim["loop"] = validate_config_sim_loop(config_sim["loop"])
    config_sim["save"] = validate_config_sim_save(config_sim["save"])

    if "sweep_vals" in config_sim:
        config_sim["sweep_vals"] = validate_config_sim_snr(config_sim["sweep_vals"])

    return config_sim



def validate_config_sim_snr(config_sim_snr):
    required_keys = {
        "start": (int, float),
        "end": (int, float),
        "step": (int, float)
    }

    validate_required_keys(config_sim_snr, required_keys, "sim.snr")

    start = config_sim_snr["start"]
    end = config_sim_snr["end"]
    step = config_sim_snr["step"]

    # Ensure 'end' is greater than or equal to 'start'
    if end < start:
        raise ValueError(f"'channel.snr.end' ({end}) must be greater than or equal to 'channel.snr.start' ({start}).")

    # Ensure 'step' is positive
    if step <= 0:
        raise ValueError(f"'channel.snr.step' ({step}) must be positive.")

    return config_sim_snr


def validate_config_ofdm(config_ofdm):
    required_keys = {
        "num_subcarriers": int,  
        "cyclic_prefix_length": int    
    }

    validate_required_keys(config_ofdm, required_keys, "ofdm")

    return config_ofdm


def validate_config_sim_loop(config_sim_loop):
    required_keys = {
        "num_frames": int,  
        "num_errors": int   
    }
    optional_keys = {
        "max_frames": (int, 1e7),  
    }

    validate_required_keys(config_sim_loop, required_keys, "sim.loop")
    validate_optional_keys(config_sim_loop, optional_keys, "sim.loop")

    num_frames = config_sim_loop["num_frames"]
    num_errors = config_sim_loop["num_errors"]
    max_frames = config_sim_loop["max_frames"]

    if num_frames < 0:
        raise ValueError(f"'sim.loop.num_frames' ({num_frames}) must be a non-negative value.")
    if num_errors < 0:
        raise ValueError(f"'sim.loop.num_errors' ({num_errors}) must be a non-negative value.")
    if max_frames < 0:
        raise ValueError(f"'sim.loop.max_frames' ({max_frames}) must be a non-negative value.")

    return config_sim_loop



def validate_config_sim_save(config_sim_save):
    optional_keys = {
        "plot_enable": (bool, False),
        "lutsim_enable": (bool, False),
        "save_output": (bool, True)
    }

    validate_optional_keys(config_sim_save, optional_keys, "sim.loop")
    
    # config_sim_save["path_output"]     = f"SC_{os.path.splitext(os.path.basename(filepath))[0]}_k{len_k}.out"
    # config_sim_save["path_fig_output"] = f"SC_{os.path.splitext(os.path.basename(filepath))[0]}_k{len_k}.png"

    return config_sim_save