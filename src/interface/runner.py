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
from src.tx.core.tx import Transmitter
from src.rx.core.rx import Receiver
from src.utils.create_run_id import create_run_id
from src.utils.output_handler import create_output_folder, save_config_to_folder
from src.utils.timekeeper import format_time

from src.utils.validation.validation_manager import validate_config

def run_polar_sim_with_len_k(
    override_config_path="configs/config_polar.json5",
    override_len_N=1024,
    override_len_k_override=512,
    override_seed=42,
    override_save_output=True,
    override_decoder_algorithm="SC",
    override_crc_enable=False,
    override_crc_length=8,
    override_quantize_enable=False,
    override_bits_chnl=5,
    override_bits_intl=6,
    override_bits_frac=1,
    override_modulation_type="QPSK",
    override_demod_type="soft",
    override_num_subcarriers=16,
    override_cyclic_prefix_length=4,
    override_sim_type="SNR",
    override_snr_start=1.0,
    override_snr_end=2.0,
    override_snr_step=1.0,
    override_num_frames=10000,
    override_num_errors=0,
    override_max_frames=10000
):
    np.random.seed(override_seed)

    polar_file_map = {
        1024: "src/lib/ecc/polar/3gpp/n1024_3gpp.pc",
        512: "src/lib/ecc/polar/3gpp/n512_3gpp.pc",
        256: "src/lib/ecc/polar/3gpp/n256_3gpp.pc",
        128: "src/lib/ecc/polar/3gpp/n128_3gpp.pc",
        64: "src/lib/ecc/polar/3gpp/n64_3gpp.pc",
        32: "src/lib/ecc/polar/3gpp/n32_3gpp.pc",
    }

    # Capture output
    terminal_output = io.StringIO()
    with redirect_stdout(terminal_output):
        # Load and modify the configuration
        config = ConfigLoader(override_config_path).get()
        config["code"]["polar"]["polar_file"] = polar_file_map[override_len_N]
        config["code"]["len_k"] = override_len_k_override
        config["code"]["polar"]["decoder"]["algorithm"] = override_decoder_algorithm
        config["code"]["polar"]["crc"]["enable"] = override_crc_enable
        config["code"]["polar"]["crc"]["length"] = override_crc_length
        config["code"]["polar"]["quantize"]["enable"] = override_quantize_enable
        config["code"]["polar"]["quantize"]["bits_chnl"] = override_bits_chnl
        config["code"]["polar"]["quantize"]["bits_intl"] = override_bits_intl
        config["code"]["polar"]["quantize"]["bits_frac"] = override_bits_frac

        # Overwrite modulation configuration
        config["mod"]["type"] = override_modulation_type
        config["mod"]["demod_type"] = override_demod_type

        # Overwrite OFDM configuration
        config["ofdm"]["num_subcarriers"] = override_num_subcarriers
        config["ofdm"]["cyclic_prefix_length"] = override_cyclic_prefix_length

        # Overwrite channel configuration
        config["sim"]["sweep_type"] = override_sim_type
        config["sim"]["sweep_vals"]["start"] = override_snr_start
        config["sim"]["sweep_vals"]["end"] = override_snr_end
        config["sim"]["sweep_vals"]["step"] = override_snr_step

        # Overwrite sim loop configuration
        config["sim"]["loop"]["num_frames"] = override_num_frames
        config["sim"]["loop"]["num_errors"] = override_num_errors
        config["sim"]["loop"]["max_frames"] = override_max_frames
        
        config = validate_config(config)

        # Create run ID and output folder
        run_id = create_run_id(config["code"]["type"], override_seed)
        output_dir = create_output_folder(run_id)
        save_config_to_folder(config, output_dir)

        # Initialize components
        code = Code(config["code"])
        channel = ChannelAWGN(config["channel"], config["sim"])
        transmitter = Transmitter(config["mod"], config["ofdm"], code)
        receiver = Receiver(config["mod"], config["ofdm"], code)
        sim = Simulation(config["sim"], output_dir)
        sim.save_output = int(override_save_output)

        len_k = code.len_k
        info_data = np.empty(len_k, dtype=np.int32)
        results = []
        status_msg, prev_status_msg = [], []
        
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
