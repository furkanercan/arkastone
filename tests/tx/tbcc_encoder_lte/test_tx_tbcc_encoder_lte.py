import pytest
from huggingface_hub import hf_hub_download
from src.tx.encoders.tbcc_encoder_lte import convolutional_encode_tail_biting_lte  # assuming your function is in encoder.py

@pytest.mark.parametrize("input_bits, expected_length", [
    ([0, 0, 0, 0, 0, 0, 0], 21),  # exactly 7 bits, minimum allowed
    ([1, 0, 1, 1, 0, 1, 0, 1], 24),  # 8 bits
    ([1]*16, 48),  # all ones
    ([0,1]*10, 60),  # alternating
])


def test_output_length(input_bits, expected_length):
    output = convolutional_encode_tail_biting_lte(input_bits)
    assert len(output) == expected_length, f"Expected {expected_length}, got {len(output)}"

def test_known_encoding_handcalc():
    # Example input of 8 bits
    u = [1, 0, 1, 1, 0, 1, 0, 0]
    # No standard expected output here, so we re-run the encoder twice to ensure determinism
    out1 = convolutional_encode_tail_biting_lte(u)
    out2 = [0,1,0,1,0,1,1,0,0,1,0,1,1,0,1,0,1,0,0,1,1,0,1,0]
    assert out1 == out2, "Encoder failed to produce expected output for known input"

def test_invalid_input_too_short():
    with pytest.raises(ValueError, match="Input must be at least as long as the constraint length"):
        convolutional_encode_tail_biting_lte([1, 0, 1])  # too short

def test_binary_output():
    u = [1, 1, 0, 1, 0, 1, 0, 1, 1]
    encoded = convolutional_encode_tail_biting_lte(u)
    assert all(bit in (0, 1) for bit in encoded), "Output must be binary"


# --- Test vector files ---
REPO_ID = "furkanercan/arkastone-test-vectors"
REPO_TYPE = "dataset"
VEC_IN_FILE = "tx/tbcc_encoder_lte/tbcc_encoder_tvec.in"
VEC_OUT_FILE = "tx/tbcc_encoder_lte/tbcc_encoder_tvec.out"

@pytest.fixture(scope="module")
def downloaded_vectors():
    vec_in_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=VEC_IN_FILE)
    vec_out_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=VEC_OUT_FILE)
    return vec_in_path, vec_out_path

def parse_bitstring(line: str) -> list:
    return [int(b) for b in line.strip() if b in ('0', '1')]

def test_tbcc_vector_matches(downloaded_vectors):
    vec_in_path, vec_out_path = downloaded_vectors

    with open(vec_in_path, "r") as f_in, open(vec_out_path, "r") as f_out:
        for idx, (in_line, out_line) in enumerate(zip(f_in, f_out)):
            input_bits = parse_bitstring(in_line)
            expected_output = parse_bitstring(out_line)

            assert len(expected_output) == 3 * len(input_bits), \
                f"Line {idx}: Expected output length {3 * len(input_bits)}, got {len(expected_output)}"

            actual_output = convolutional_encode_tail_biting_lte(input_bits)

            assert actual_output == expected_output, \
                f"Line {idx}: Output mismatch\nExpected: {expected_output}\nActual:   {actual_output}"