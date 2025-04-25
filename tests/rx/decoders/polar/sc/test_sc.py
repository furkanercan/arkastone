import numpy as np
from src.coding.coding import Code
from src.configs.config_code import CodeConfig
from src.utils.validation.config_validator import validate_config_code
from src.rx.decoders.polar.sc import PolarDecoder_SC

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


def test_decoder_initialization():
    # Initialize decoder
    decoder = PolarDecoder_SC(code)

    # Test attribute initialization
    assert np.array_equal(decoder.vec_polar_isfrozen, code.frozen_bits)
    assert decoder.len_logn     == code.code.len_logn
    assert decoder.qtz_enable   == code.code.quantize.enable
    assert decoder.qtz_int_max  == code.code.quantize.intl_max
    assert decoder.qtz_int_min  == code.code.quantize.intl_min



def test_create_decoding_schedule():
    # Initialize the decoder
    decoder = PolarDecoder_SC(code)

    # Test schedule creation
    decoder.initialize_decoder()
    assert len(decoder.vec_dec_sch) > 0, "Decoding schedule should not be empty."
    assert len(decoder.vec_dec_sch_size) == len(decoder.vec_dec_sch), "Size vector length mismatch."
    assert len(decoder.vec_dec_sch_depth) == len(decoder.vec_dec_sch), "Depth vector length mismatch."
    assert len(decoder.vec_dec_sch_dir) == len(decoder.vec_dec_sch), "Direction vector length mismatch."

def test_dec_sc_f():
    # Initialize the decoder
    decoder = PolarDecoder_SC(code)
    decoder.initialize_decoder()

    # Example inputs for dec_sc_f
    stage_size = 4
    stage_depth = 2
    decoder.mem_alpha[stage_depth] = np.array([2.0, -3.0, 1.0, -1.0])  # Set example alpha values

    # Perform F-node computation
    decoder.dec_sc_f(stage_size, stage_depth, is_quantized=True, max_value=code.code.quantize.intl_max)

    # Check the results
    expected_result = np.array([1.0, 1.0])  # Expected output after F-node computation
    np.testing.assert_array_almost_equal(
        decoder.mem_alpha[stage_depth - 1][:stage_size // 2],
        expected_result,
        decimal=1,
        err_msg="F-node computation failed."
    )



def test_dec_sc():
    # I/O
    vec_llr = np.random.randn(2**code.len_logn)  # Random LLR values
    vec_decoded = np.empty(code.len_k, dtype=bool)

    # Initialize the decoder
    decoder = PolarDecoder_SC(code)
    decoder.initialize_decoder()


    # Test decoding
    decoder.dec_sc(vec_decoded, vec_llr)

    # Check the length of the decoded vector
    num_info_bits = sum(1 for bit in code.code.frozen_bits if bit == 0)
    assert len(vec_decoded) == num_info_bits, "Decoded vector length mismatch."
