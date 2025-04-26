import numpy as np
from pathlib import Path
from src.coding.coding import Code
from src.configs.config_code import CodeConfig
from src.utils.validation.config_validator import validate_config_code
from src.rx.decoders.polar.sc import PolarDecoder_SC

# Setup code config
code_config = {
    "type": "POLAR",
    "len_k": 128,
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



def test_dec_sc_testvecs():
    # File paths
    base_name = "ppile_sc_n1024_k128_3gpp_Q0"  # Example
    base_path = Path("tests/rx/decoders/polar/testvectors_ppile/sc/processed")
    llr_file = base_path / f"{base_name}.in"
    ref_file = base_path / f"{base_name}.out"

    # Load test vectors
    with open(llr_file, "r") as f:
        llr_lines = [list(map(float, line.strip().split())) for line in f.readlines()]

    with open(ref_file, "r") as f:
        ref_lines = [list(map(int, line.strip().split())) for line in f.readlines()]

    assert len(llr_lines) == len(ref_lines), "Mismatch in number of LLRs and decoded vectors"

    decoder = PolarDecoder_SC(code)
    decoder.initialize_decoder()

    total = len(llr_lines)
    passed = 0

    for idx, (llr_vec, ref_vec) in enumerate(zip(llr_lines, ref_lines)):
        vec_llr = np.array(llr_vec)
        vec_decoded = np.empty(code.len_k, dtype=bool)

        decoder.dec_sc(vec_decoded, vec_llr)

        if np.array_equal(vec_decoded.astype(int), ref_vec):
            passed += 1
        else:
            assert np.array_equal(vec_decoded, ref_vec), (
                f"\n❌ Mismatch at vector {idx}:\n"
                f"Expected: {ref_vec}\n"
                f"Got     : {vec_decoded.tolist()}"
            )


    print(f"\n✅ Passed {passed}/{total} tests.")
