import numpy as np
import math
import string
import time

from src.coding.coding import Code
from src.tx.tx import Transmitter
from src.rx.rx import Receiver
from src.sim.sim import Simulation
from src.channel.awgn import ChannelAWGN
from src.utils.validation.config_loader import ConfigLoader
from src.utils.output_handler import *
from src.utils.create_run_id import *
from src.utils.timekeeper import *


seed = 42
np.random.seed(seed)
config_file = "configs/config_polar.json5"
config = ConfigLoader(config_file).get()
run_id = create_run_id(config["code"]["type"], seed) #Make part of sim_config
output_dir = create_output_folder(run_id) #Make part of sim_config
save_config_to_folder(config, output_dir)

code_config = config["code"]
channel_config = config["channel"]
mod_config = config["mod"]
sim_config = config["sim"]
ofdm_config = config["ofdm"]

sim = Simulation(sim_config, output_dir)
code = Code(code_config) 
transmitter = Transmitter(mod_config, ofdm_config, code)
channel = ChannelAWGN(channel_config, seed)
receiver = Receiver(mod_config, ofdm_config, code)

len_k = code.len_k
status_msg, prev_status_msg = [], []

info_data = np.empty(len_k, dtype=np.int32) 

for idx, (stdev, var) in enumerate(zip(channel.stdev, channel.variance)):
    time_start = time.time()
    snr_point = config["channel"]["snr"]["simpoints"]
    while(sim.run_simulation(idx)):
        info_data[:] = np.random.randint(0, 2, size=len_k)
        # info_data[:] = np.zeros(len_k, dtype=np.int32)
        transmitter.tx_chain(info_data)
        received_data = channel.apply_awgn(transmitter.transmitted_data, stdev, var)
        receiver.rx_chain(received_data, var)
        sim.collect_run_stats(idx, 1023, 1, info_data, receiver.decoded_data)

        if(sim.count_frame[idx] % 100 == 0):
            time_end = time.time()
            time_elapsed = time_end - time_start
            sim.update_run_results(idx, len_k)
            status_msg = sim.display_run_results_temp(idx, snr_point[idx], format_time(time_elapsed), prev_status_msg)
            prev_status_msg = status_msg

    time_end = time.time()
    time_elapsed = time_end - time_start
    sim.update_run_results(idx, len_k)
    status_msg = sim.display_run_results_perm(idx, snr_point[idx], format_time(time_elapsed), prev_status_msg)
    prev_status_msg = status_msg