import numpy as np
import time
import io
import sys
import json5
from contextlib import redirect_stdout

from src.utils.validation.config_loader import ConfigLoader
from src.sim.sim import Simulation
from src.channel.awgn import ChannelAWGN
from src.coding.coding import Code
from src.tx.tx import Transmitter
from src.rx.rx import Receiver
from src.utils.create_run_id import create_run_id
from src.utils.output_handler import create_output_folder, save_config_to_folder
from src.utils.timekeeper import format_time

def run_polar_sim_with_len_k(
    override_config_path="configs/config_polar.json5",
    override_len_k_override=512,
    override_seed=42,
    override_save_output=True,
    override_decoder_algorithm="SC",
    override_crc_enable=False,
    override_crc_length=8,
    override_quantize_enable=False,
    override_bits_chnl=5,
    override_bits_intl=6,
    override_bits_frac=1
):
    np.random.seed(seed)

    # Capture output
    terminal_output = io.StringIO()
    with redirect_stdout(terminal_output):
        # Load and modify the configuration
        config = ConfigLoader(override_config_path).get()
        config["code"]["len_k"] = override_len_k_override  # Inject user value

        # Overwrite decoder algorithm
        config["code"]["polar"]["decoder"]["algorithm"] = override_decoder_algorithm

        # Overwrite CRC configuration
        config["code"]["polar"]["crc"]["enable"] = override_crc_enable
        config["code"]["polar"]["crc"]["length"] = override_crc_length

        # Overwrite quantization configuration
        config["code"]["polar"]["quantize"]["enable"] = override_quantize_enable
        config["code"]["polar"]["quantize"]["bits_chnl"] = override_bits_chnl
        config["code"]["polar"]["quantize"]["bits_intl"] = override_bits_intl
        config["code"]["polar"]["quantize"]["bits_frac"] = override_bits_frac

        # Create run ID and output folder
        run_id = create_run_id(config["code"]["type"], override_seed)
        output_dir = create_output_folder(run_id)
        save_config_to_folder(config, output_dir)

        # Initialize components
        code = Code(config["code"])
        channel = ChannelAWGN(config["channel"], override_seed)
        transmitter = Transmitter(config["mod"], config["ofdm"], code)
        receiver = Receiver(config["mod"], config["ofdm"], code)
        sim = Simulation(config["sim"], output_dir)
        sim.save_output = int(override_save_output)  # Assuming your class uses int(1/0)

        len_k = code.len_k
        info_data = np.empty(len_k, dtype=np.int32)
        results = []
        status_msg, prev_status_msg = [], []

        for idx, (stdev, var) in enumerate(zip(channel.stdev, channel.variance)):
            time_start = time.time()
            snr_point = config["channel"]["snr"]["simpoints"]
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

            time_end = time.time()
            time_elapsed = time_end - time_start
            sim.update_run_results(idx, len_k)
            status_msg = sim.display_run_results_perm(idx, snr_point[idx], format_time(time_elapsed), prev_status_msg)
            prev_status_msg = status_msg
            res = sim.get_ber_results(idx, len_k)
            res.update({"snr": snr_point[idx], "time": format_time(time_elapsed)})
            results.append(res)

    output_text = terminal_output.getvalue()
    return results, output_text
