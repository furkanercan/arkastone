import numpy as np
# from src.common.create_polar_indices import create_polar_indices
from src.tx.encoders.polar_encoder import PolarEncoder
from src.coding.coding import Code
from src.configs.config_code import CodeConfig
from src.utils.validation.config_validator import validate_config_code

def test_polar_encoder():
    #Initialize code
    code_config = {
        "type": "POLAR",
        "len_k": 512,
        "polar":{
            "polar_file": "src/lib/ecc/polar/3gpp/n1024_3gpp.pc",
            "crc":{
                "enable": False,
                "name": "my_crc",
                "length" : 6,
                "preload_val": 0,
                "mode": "generic"
            },
            "decoder":{
                "algorithm": "SC",
                "flip_max_iters": 30,
                "list_size": 8
            },
            "quantize": {
                "enable": False,
                "bits_chnl": 5,
                "bits_intl": 6,
                "bits_frac": 1
            },
            "fast_mode": {
                "enable": True,
                "max_rate0": 1024,
                "max_rate1": 1024,
                "max_rep": 1024,
                "max_spc": 1024,
                "max_ml_0011": 0,
                "max_ml_0101": 0
            }
        }
    }

    validate_config_code(code_config)
    config_code = CodeConfig.from_dict(code_config)
    code = Code(config_code) 

    # Initialize test variables
    uncoded_data = np.random.randint(0, 2, size=code.len_k)
    encoded_data = np.empty(code.code.len_n, dtype=bool)
    
    # Instantiate and call class
    encoder = PolarEncoder(code)
    encoded_data = encoder.encode(uncoded_data)
    
    # Test the outcome
    assert len(encoded_data) == code.code.len_n  # Block length for len_logn=3
    assert (encoded_data == (np.array(uncoded_data) @ encoder.matG_kxN) % 2).all()

    matrices = encoder.export_matrices()
    assert matrices["matG_NxN"].shape == (code.code.len_n, code.code.len_n)
    assert matrices["matG_kxN"].shape == (code.len_k, code.code.len_n)
    assert matrices["matHt"].shape == (code.code.len_n, code.len_k)  


# test_polar_encoder()