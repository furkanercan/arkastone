from src.configs.config_code import CodeConfig
from src.configs.config_modulation import ModConfig
from src.configs.config_channel import ChannelConfig
from src.configs.config_ofdm import OFDMConfig
from src.configs.config_sim import SimConfig, SimSweepVals, SimLoopConfig, SimSaveConfig

from src.coding.coding import Code
from src.tx.core.tx import Transmitter
from src.rx.core.rx import Receiver
from src.channel.awgn import ChannelAWGN
from src.sim.sim import Simulation
from src.utils.output_handler import *
from src.utils.create_run_id import *
from src.utils.timekeeper import *
import numpy as np
import time

def run_simulation_from_config(config: dict, progress_callback=None) -> dict:
    run_id = create_run_id(config["code"]["type"], config["channel"]["seed"]) #Make part of config_sim
    output_dir = create_output_folder(run_id) #Make part of sim_config
    save_config_to_folder(config, output_dir)

    config_code = CodeConfig.from_dict(config["code"])
    config_chn = ChannelConfig(**config["channel"])
    config_mod = ModConfig(**config["mod"])
    config_sim = SimConfig(
        mode=config["sim"]["mode"],
        sweep_type=config["sim"]["sweep_type"],
        sweep_vals=SimSweepVals(**config["sim"]["sweep_vals"]),
        loop=SimLoopConfig(**config["sim"]["loop"]),
        save=SimSaveConfig(**config["sim"]["save"])
    )
    config_ofdm = OFDMConfig(**config["ofdm"])

    

    sim = Simulation(config_sim, output_dir)
    code = Code(config_code) 
    transmitter = Transmitter(config_mod, config_ofdm, code)
    channel = ChannelAWGN(config_chn, config_sim)
    receiver = Receiver(config_mod, config_ofdm, code)

    len_k = code.len_k
    status_msg, prev_status_msg = [], []

    info_data = np.empty(len_k, dtype=np.int32) 
    for idx, (stdev, var) in enumerate(zip(channel.stdev, channel.variance)):
        time_start = time.time()
        while(sim.run_simulation(idx)):
            info_data[:] = np.random.randint(0, 2, size=len_k)
            transmitter.tx_chain(info_data)
            received_data = channel.apply_awgn(transmitter.transmitted_data, stdev, var)
            receiver.rx_chain(received_data, var)
            sim.collect_run_stats(idx, 1023, 1, info_data, receiver.decoded_data)

            if(sim.count_frame[idx] % 100 == 0):
                time_end = time.time()
                time_elapsed = time_end - time_start
                sim.update_run_results(idx, len_k)
                status_msg = sim.display_run_results_temp(idx, sim.simpoints[idx], format_time(time_elapsed), prev_status_msg)
                prev_status_msg = status_msg
                if progress_callback:
                    progress_callback({
                        "snr_point": sim.simpoints[idx],
                        "ber": sim.ber[idx],
                        "bler": sim.bler[idx],
                        # "avg_steps": sim.avg_steps[idx],
                        "avg_iters": sim.avg_iters[idx],
                        "frames": sim.count_frame[idx],
                        "errors": sim.count_frame_error[idx],
                        "time_elapsed": format_time(time_elapsed),
                        "type": "temp" 
                    })

        time_end = time.time()
        time_elapsed = time_end - time_start
        sim.update_run_results(idx, len_k)
        status_msg = sim.display_run_results_perm(idx, sim.simpoints[idx], format_time(time_elapsed), prev_status_msg)
        prev_status_msg = status_msg
        if progress_callback:
            progress_callback({
                "snr_point": sim.simpoints[idx],
                "ber": sim.ber[idx],
                "bler": sim.bler[idx],
                # "avg_steps": sim.avg_steps[idx],
                "avg_iters": sim.avg_iters[idx],
                "frames": sim.count_frame[idx],
                "errors": sim.count_frame_error[idx],
                "time_elapsed": format_time(time_elapsed),
                "type": "perm" 
            })

    return {
        "status": "done",
        "snr_point": sim.simpoints,
        "ber": sim.ber,
        "bler": sim.bler,
        # "avg_steps": sim.avg_steps[idx],
        "avg_iters": sim.avg_iters,
        "frames": sim.count_frame,
        "errors": sim.count_frame_error,
        "type": "perm" 
    }