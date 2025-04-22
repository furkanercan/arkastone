import numpy as np
import math
import string
import time

from src.coding.coding import Code
from src.sim.sim import Simulation
from src.configs.config_sim import SimConfig, SimSweepVals, SimLoopConfig, SimSaveConfig
from src.channel.awgn import ChannelAWGN
from src.configs.config_channel import ChannelConfig
from src.utils.validation.config_loader import ConfigLoader
from src.utils.output_handler import *
from src.utils.create_run_id import *
from src.utils.timekeeper import *

from src.configs.config_modulation import ModConfig
from src.tx.encoders.encoder import UncodedEncoder
from src.tx.core.modulator import Modulator
from src.rx.core.demodulator import Demodulator

from src.coding.coding import Code
from src.configs.config_code import CodeConfig
from src.rx.decoders.uncoded_decoder import UncodedDecoder

def parse_reference_data(config):
    """
    Parse reference data (SNR, BER, BLER) from the config dictionary.
    The data should be in scientific notation as lists of floats.
    """
    reference_data = config.get("reference", {})

    snr = reference_data.get("snr", [])
    ber = reference_data.get("ber", [])
    bler = reference_data.get("bler", [])

    # Convert to floating-point numbers, ensuring correct parsing of scientific notation
    snr = [float(val) for val in snr]
    ber = [float(val) for val in ber]
    bler = [float(val) for val in bler]

    return snr, ber, bler

def main_test_uncoded(config_file):
    configloader = ConfigLoader(config_file)
    config = configloader.get()
    ref_snr, ref_ber, ref_bler = parse_reference_data(configloader.raw_config)

    # code_config    = config["code"]
    config_code = CodeConfig.from_dict(config["code"])
    channel_config = ChannelConfig(**config["channel"])
    mod_config     = ModConfig(**config["mod"])
    sim_config     = SimConfig(
        mode=config["sim"]["mode"],
        sweep_type=config["sim"]["sweep_type"],
        sweep_vals=SimSweepVals(**config["sim"]["sweep_vals"]),
        loop=SimLoopConfig(**config["sim"]["loop"]),
        save=SimSaveConfig(**config["sim"]["save"])
    )

    sim         = Simulation(sim_config)
    code        = Code(config_code) 
    encoder     = UncodedEncoder()
    modulator   = Modulator(mod_config)
    channel     = ChannelAWGN(channel_config, sim_config)
    demodulator = Demodulator(mod_config)
    decoder     = UncodedDecoder()
    decoder.initialize_decoder()

    len_k = code.len_k
    len_n = code.len_k # N = k for uncoded
    status_msg, prev_status_msg = [], []

    if modulator.scheme == "bpsk":
        modulated_data = np.empty(len_n, dtype=int)  # Real values for BPSK
    else:
        modulated_data = np.empty(len_n // modulator.log_num_constellations, dtype=complex)  # Complex values for QPSK
    
    info_data      = np.empty(len_k, dtype=np.int32) 
    encoded_data   = np.empty(len_n, dtype=np.int32) 
    
    vec_llr        = np.empty(len_n, dtype=np.float32)
    decoded_data   = np.empty(len_k, dtype=np.bool)

    tolerance = 1e-6 # Reference doesn't have resolution past 1e-6

    for idx, (stdev, var) in enumerate(zip(channel.stdev, channel.variance)):
        time_start = time.time()
        while(sim.run_simulation(idx)):
            info_data[:] = np.random.randint(0, 2, size=len_k)
            encoded_data = encoder.encode(info_data)
            modulator.modulate(modulated_data, encoded_data)
            received_data = channel.apply_awgn(modulated_data, stdev, var)
            demodulator.demodulate(vec_llr, received_data, var)
            decoder.decode_chain(decoded_data, vec_llr)
            sim.collect_run_stats(idx, 1023, 1, info_data, decoded_data)

            if(sim.count_frame[idx] % 100 == 0):
                time_end = time.time()
                time_elapsed = time_end - time_start
                sim.update_run_results(idx, len_k)
                status_msg = sim.display_run_results_temp(idx, sim.simpoints[idx], format_time(time_elapsed), prev_status_msg)
                prev_status_msg = status_msg

        time_end = time.time()
        time_elapsed = time_end - time_start   
        sim.update_run_results(idx, len_k)

        # Validate against the reference data with tolerance
        assert abs(sim.simpoints[idx] - ref_snr[idx])  <= tolerance, f"Failure test_uncoded: {config_file}: SNR mismatch at point {sim.simpoints[idx]} (expected {ref_snr[idx]})"
        assert abs(sim.ber[idx]   - ref_ber[idx])  <= tolerance, f"Failure test_uncoded: {config_file}: BER mismatch at point {sim.simpoints[idx]} dB: got {sim.ber[idx]}, (expected {ref_ber[idx]})"
        assert abs(sim.bler[idx]  - ref_bler[idx]) <= tolerance, f"Failure test_uncoded: {config_file}: BLER mismatch at point {sim.simpoints[idx]} dB: got {sim.bler[idx]} (expected {ref_bler[idx]})"

        
        status_msg = sim.display_run_results_perm(idx, sim.simpoints[idx], format_time(time_elapsed), prev_status_msg)
        prev_status_msg = status_msg


def test_uncoded():
    config_files = [
        "tests/endtoend/decoding/uncoded/config_test_uncoded_modBPSK_chnAWGN_n128.json5", 
        "tests/endtoend/decoding/uncoded/config_test_uncoded_modQPSK_chnAWGN_n128.json5", 
        "tests/endtoend/decoding/uncoded/config_test_uncoded_mod16QAM_chnAWGN_n128.json5"
    ]

    for config_file in config_files:
        main_test_uncoded(config_file)


# #Uncomment the following for individual testing or debugging. 
# Comment when pushing or pytest'ing.
# test_uncoded() 