import numpy as np
from pathlib import Path
from src.coding.coding import Code
from src.configs.config_code import CodeConfig
from src.utils.validation.config_validator import validate_config_code
from src.rx.decoders.polar.sc import PolarDecoder_SC

# Setup code config
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


def test_dec_sc_testvecs():
    base_name = "ppile_fastssc_n1024_k512_3gpp_Q0"
    base_path = Path("tests/rx/decoders/polar/testvectors_ppile/fastssc/processed")
    
    # === Load LLRs and reference outputs ===
    llr_file = base_path / f"{base_name}.in"
    ref_file = base_path / f"{base_name}.out"

    with open(llr_file, "r") as f:
        llr_lines = [list(map(float, line.strip().split())) for line in f]

    with open(ref_file, "r") as f:
        ref_lines = [list(map(int, line.strip().split())) for line in f]

    assert len(llr_lines) == len(ref_lines), "Mismatch in number of LLRs and decoded vectors"

    # === Load and compare schedule only once ===
    sch_file = base_path / f"{base_name}.sch"
    size_file = base_path / f"{base_name}.size"
    dir_file = base_path / f"{base_name}.dir"

    with open(sch_file, "r") as f:
        vec_dec_sch = f.read().strip().split()

    with open(size_file, "r") as f:
        vec_dec_sch_size = list(map(int, f.read().strip().split()))

    with open(dir_file, "r") as f:
        vec_dec_sch_dir = list(map(int, f.read().strip().split()))

    # Compare with generated schedule once
    decoder = PolarDecoder_SC(code)
    decoder.initialize_decoder()

    # if vec_dec_sch != decoder.vec_dec_sch:
    #     mismatch_file = Path("schedule_mismatch_vec_dec_sch.txt")
    #     with open(mismatch_file, "w") as f:
    #         f.write("Expected Schedule:\n")
    #         f.write(" ".join(vec_dec_sch) + "\n")
    #         f.write("Got Schedule:\n")
    #         f.write(" ".join(decoder.vec_dec_sch) + "\n")
    #         mismatch_indices = [i for i, (a, b) in enumerate(zip(vec_dec_sch, decoder.vec_dec_sch)) if a != b]
    #         f.write("Mismatch Indices:\n")
    #         f.write(" ".join(map(str, mismatch_indices)) + "\n")
    #     assert vec_dec_sch == decoder.vec_dec_sch, (
    #         f"❌ Schedule instruction mismatch. Mismatch saved to: {mismatch_file}"
    #     )

    # if vec_dec_sch_size != decoder.vec_dec_sch_size:
    #     mismatch_file = Path("schedule_mismatch_vec_dec_sch_size.txt")
    #     with open(mismatch_file, "w") as f:
    #         f.write("Expected Schedule Sizes:\n")
    #         f.write(" ".join(map(str, vec_dec_sch_size)) + "\n")
    #         f.write("Got Schedule Sizes:\n")
    #         f.write(" ".join(map(str, decoder.vec_dec_sch_size)) + "\n")
    #         mismatch_indices = [i for i, (a, b) in enumerate(zip(vec_dec_sch_size, decoder.vec_dec_sch_size)) if a != b]
    #         f.write("Mismatch Indices:\n")
    #         f.write(" ".join(map(str, mismatch_indices)) + "\n")
    #     assert vec_dec_sch_size == decoder.vec_dec_sch_size, (
    #         f"❌ Schedule stage size mismatch. Mismatch saved to: {mismatch_file}"
    #     )

    # if vec_dec_sch_dir != decoder.vec_dec_sch_dir:
    #     mismatch_file = Path("schedule_mismatch_vec_dec_sch_dir.txt")
    #     with open(mismatch_file, "w") as f:
    #         f.write("Expected Schedule Directions:\n")
    #         f.write(" ".join(map(str, vec_dec_sch_dir)) + "\n")
    #         f.write("Got Schedule Directions:\n")
    #         f.write(" ".join(map(str, decoder.vec_dec_sch_dir)) + "\n")
    #         mismatch_indices = [i for i, (a, b) in enumerate(zip(vec_dec_sch_dir, decoder.vec_dec_sch_dir)) if a != b]
    #         f.write("Mismatch Indices:\n")
    #         f.write(" ".join(map(str, mismatch_indices)) + "\n")
    #     assert vec_dec_sch_dir == decoder.vec_dec_sch_dir, (
    #         f"❌ Schedule direction mismatch. Mismatch saved to: {mismatch_file}"
    #     )

    # === Decode all vectors ===
    total = len(llr_lines)
    passed = 0

    for idx, (llr_vec, ref_vec) in enumerate(zip(llr_lines, ref_lines)):
        vec_llr = np.array(llr_vec)
        vec_decoded = np.empty(code.len_k, dtype=bool)

        decoder.dec_sc(vec_decoded, vec_llr)

        if np.array_equal(vec_decoded.astype(int), ref_vec):
            passed += 1
        else:
            mismatch_indices = [i for i, (a, b) in enumerate(zip(ref_vec, vec_decoded.astype(int))) if a != b]
            mismatch_file = Path(f"mismatch_vector_{idx}.txt")
            with open(mismatch_file, "w") as f:
                f.write("Expected:\n")
                f.write(" ".join(map(str, ref_vec)) + "\n")
                f.write("Got:\n")
                f.write(" ".join(map(str, vec_decoded.astype(int))) + "\n")
                f.write("Mismatch Indices:\n")
                f.write(" ".join(map(str, mismatch_indices)) + "\n")

            assert np.array_equal(vec_decoded, ref_vec), (
                f"\n❌ Mismatch at vector {idx}:\n"
                f"Expected: {ref_vec}\n"
                f"Got     : {vec_decoded.tolist()}\n"
                f"Mismatch Indices: {mismatch_indices}\n"
                f"Mismatch saved to: {mismatch_file}"
            )

    print(f"\n✅ Passed {passed}/{total} vectors.")