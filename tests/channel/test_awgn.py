import numpy as np
import random
from src.channel.awgn import ChannelAWGN
from src.configs.config_channel import ChannelConfig
from src.configs.config_sim import SimConfig, SimSweepVals, SimLoopConfig, SimSaveConfig
from src.utils.validation.config_validator import validate_config_channel, validate_config_sim

from src.configs.config_modulation import ModConfig
from src.tx.core.modulator import Modulator
from src.utils.validation.config_validator import validate_config_modulator

def test_awgn_channel_real():
    config_sim = {
        "loop": {
            "num_frames": 10000,
            "num_errors": 50,
            "max_frames": 10000000
        },
        "save": {
            "plot_enable": False,
            "lutsim_enable": False,
            "save_output": False
        },
        "mode": "dev",
        "sweep_type": "SNR",
        "sweep_vals": {
            "start": -2,
            "end": 15,
            "step": 0.5
        }
    }
    config_chn = {
        "type": "AWGN",
        "seed": 42
    }

    validate_config_channel(config_chn)
    validate_config_sim(config_sim)

    chn_config = ChannelConfig(**config_chn)
    sim_config = SimConfig(
        mode=config_sim["mode"],
        sweep_type=config_sim["sweep_type"],
        sweep_vals=SimSweepVals(**config_sim["sweep_vals"]),
        loop=SimLoopConfig(**config_sim["loop"]),
        save=SimSaveConfig(**config_sim["save"])
    )

    for _ in range(5):  
        vec_mod = np.random.choice([-1, 1], size=10000000)
        stdev = 0.35
        variance = stdev**2
        channel = ChannelAWGN(chn_config, sim_config) 
        vec_awgn = channel.apply_awgn(vec_mod, stdev, variance)

        noise = vec_awgn - vec_mod

        # Assertions to check if the noise characteristics are as expected
        assert np.isclose(noise.mean(), 0, atol=1e-1), "Mean of noise is not close to 0."
        assert np.isclose(noise.std(), stdev, atol=1e-2), f"Stdev of noise is not close to {stdev}."

    

def test_awgn_channel_complex():
    config_sim = {
        "loop": {
            "num_frames": 10000,
            "num_errors": 50,
            "max_frames": 10000000
        },
        "save": {
            "plot_enable": False,
            "lutsim_enable": False,
            "save_output": False
        },
        "mode": "dev",
        "sweep_type": "SNR",
        "sweep_vals": {
            "start": -2,
            "end": 15,
            "step": 0.5
        }
    }
    config_chn = {
        "type": "AWGN",
        "seed": 42
    }

    mod_config_qpsk = ModConfig(type='qpsk', demod_type='hard')
    mod_config_16qam = ModConfig(type='16qam', demod_type='hard')
    # mod_config_64qam = ModConfig(type='64qam', demod_type='hard')

    mod_dict = {"4":  mod_config_qpsk,
                "16": mod_config_16qam} #Include future mods here later.

    validate_config_channel(config_chn)
    validate_config_sim(config_sim)

    chn_config = ChannelConfig(**config_chn)
    sim_config = SimConfig(
        mode=config_sim["mode"],
        sweep_type=config_sim["sweep_type"],
        sweep_vals=SimSweepVals(**config_sim["sweep_vals"]),
        loop=SimLoopConfig(**config_sim["loop"]),
        save=SimSaveConfig(**config_sim["save"])
    )

    for _ in range(10):  # Run the test 100 times
        mod_type_key = random.choice(list(mod_dict.keys()))
        mod_config = mod_dict[mod_type_key]

        # validate_config_modulator(mod_config)
        modulator = Modulator(mod_config)
        
        vec_size = 1000000
        vec_bool = np.random.choice([0, 1], size=vec_size)
        vec_mod = np.empty(int(vec_size/modulator.log_num_constellations), dtype=complex)

        modulator.modulate(vec_mod, vec_bool)
        
        stdev = random.uniform(0.1, 1.0)
        variance = stdev**2
        channel = ChannelAWGN(chn_config, sim_config)  # Pass the config_chn to ChannelAWGN
        vec_awgn = channel.apply_awgn(vec_mod, stdev, variance)

        noise = vec_awgn - vec_mod

        # Assertions to check if the noise characteristics are as expected
        assert np.isclose(noise.mean(), 0, atol=1e-2), "Mean of noise is not close to 0."
        assert np.isclose(noise.std(), stdev, atol=1e-2), f"Stdev of noise is not close to {stdev}."
