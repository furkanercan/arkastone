import numpy as np
# from src.common.create_polar_indices import create_polar_indices
from src.tx.tx import Transmitter
from src.coding.coding import Code
from src.utils.validation.config_validator import validate_config_code

#Initialize code
code_config = {
    "type": "POLAR",
    "len_k": 512,
    "polar":{
        "polar_file": "src/lib/ecc/polar/3gpp/n1024_3gpp.pc",
        "crc":{
            "enable": False,
            "length" : 8
        },
        "decoder":{
            "algorithm": "SC",
            "flip_max_iters": 30
        },
        "quantize": {
            "enable": False,
            "bits_chnl": 5,
            "bits_intl": 6,
            "bits_frac": 1
        },
        "fast_enable": True,
        "fast_max_size": {
            "rate0": 1024,
            "rate1": 1024,
            "rep": 1024,
            "spc": 1024,
            "ml_0011": 0,
            "ml_0101": 0
        }
    }
}

mod_config = {
    "type": "16QAM",
    "demod_type": "soft"

}

ofdm_config = {
    "num_subcarriers": 16,
    "cyclic_prefix_length": 4
}

validate_config_code(code_config)
code = Code(code_config) 

def test_transmitter():
    # Initialize test variables
    uncoded_data = np.random.randint(0, 2, size=code.len_k)
    
    # Instantiate and call class
    transmitter = Transmitter(mod_config, ofdm_config, code)
    transmitter.tx_chain(uncoded_data)
    
    # Test the outcome
    assert len(transmitter.encoded_data) == code.len_n  # Block length for len_logn=3
    # assert (transmitted_data == (np.array(uncoded_data) @ transmitter.encoder.matG_kxN) % 2).all()

    matrices = transmitter.encoder.export_matrices()
    assert matrices["matG_NxN"].shape == (code.len_n, code.len_n)
    assert matrices["matG_kxN"].shape == (code.len_k, code.len_n)
    assert matrices["matHt"].shape == (code.len_n, code.len_k)  # Assuming 4 non-info bits
