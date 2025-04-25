import numpy as np
import time
import io
import sys
import json5
from contextlib import redirect_stdout

from src.sim.sim import Simulation
from src.channel.awgn import ChannelAWGN
from src.coding.coding import Code
from src.tx.core.tx import Transmitter
from src.rx.core.rx import Receiver
from src.utils.create_run_id import create_run_id
from src.utils.output_handler import create_output_folder, save_config_to_folder
from src.utils.timekeeper import format_time

def streamlit_runner(config: dict):
    # Capture output
    terminal_output = io.StringIO()
    with redirect_stdout(terminal_output):

        run_id = create_run_id(config["code"]["type"], config["channel"]["seed"]) #Make part of config_sim
        output_dir = create_output_folder(run_id) #Make part of sim_config
        save_config_to_folder(config, output_dir)

        config_code = config["code"]
        config_chn  = config["channel"]
        config_mod  = config["mod"]
        config_sim  = config["sim"]
        config_ofdm = config["ofdm"]

        sim = Simulation(config_sim, output_dir)
        code = Code(config_code) 
        transmitter = Transmitter(config_mod, config_ofdm, code)
        channel = ChannelAWGN(config_chn, config_sim)
        receiver = Receiver(config_mod, config_ofdm, code)

        len_k = code.len_k
        status_msg, prev_status_msg = [], []

        info_data = np.empty(len_k, dtype=np.int32) 
        results = []
       
        for idx, (stdev, var) in enumerate(zip(channel.stdev, channel.variance)):
            time_start = time.time()
            while sim.run_simulation(idx):
                info_data[:] = np.random.randint(0, 2, size=len_k)
                transmitter.tx_chain(info_data)
                received_data = channel.apply_awgn(transmitter.transmitted_data, stdev, var)
                receiver.rx_chain(received_data, var)
                sim.collect_run_stats(idx, 1023, 1, info_data, receiver.decoded_data)

                if sim.count_frame[idx] % 100 == 0:
                    time_end = time.time()
                    time_elapsed = time_end - time_start
                    sim.update_run_results(idx, len_k)
                    res = sim.get_ber_results(idx, len_k)
                    res.update({"snr": sim.simpoints[idx], "time": format_time(time_elapsed)})
                    if idx < len(results):
                        results[idx] = res
                    else:
                        results.append(res)

                    # Yield intermediate results
                    yield results, terminal_output.getvalue()

            time_end = time.time()
            time_elapsed = time_end - time_start
            sim.update_run_results(idx, len_k)
            status_msg = sim.display_run_results_perm(idx, sim.simpoints[idx], format_time(time_elapsed), prev_status_msg)
            prev_status_msg = status_msg
            res = sim.get_ber_results(idx, len_k)
            res.update({"snr": sim.simpoints[idx], "time": format_time(time_elapsed)})
            results.append(res)
            # results[idx] = res # update the existing entry instead of appending

            # Yield final results for this SNR point
            yield results, terminal_output.getvalue()

    output_text = terminal_output.getvalue()
    yield results, output_text
