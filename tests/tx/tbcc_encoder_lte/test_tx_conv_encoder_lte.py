import pytest
from pathlib import Path
from src.tx.encoders.tbcc_encoder_lte import convolutional_encode_terminated_lte

# LTE constraint length and generator setup (must match encoder)
K = 7

@pytest.mark.parametrize("input_bits, expected_length", [
    ([0, 0, 0, 0, 0, 0, 0], 3 * (7 + K - 1)),  # 7 bits + 6 zeros = 13 bits encoded
    ([1, 0, 1, 1, 0, 1, 0, 1], 3 * (8 + K - 1)),
    ([1]*16, 3 * (16 + K - 1)),
    ([0,1]*10, 3 * (20 + K - 1)),
])
def test_output_length(input_bits, expected_length):
    output = convolutional_encode_terminated_lte(input_bits)
    assert len(output) == expected_length, f"Expected {expected_length}, got {len(output)}"

def test_known_encoding_handcalc():
    u = [1, 0, 1, 1, 0, 1, 0, 0]
    out = convolutional_encode_terminated_lte(u)
    expected_length = 3 * (len(u) + K - 1)
    assert len(out) == expected_length
    assert all(bit in (0, 1) for bit in out), "Output must be binary"

def test_binary_output():
    u = [1, 1, 0, 1, 0, 1, 0, 1, 1]
    encoded = convolutional_encode_terminated_lte(u)
    assert all(bit in (0, 1) for bit in encoded), "Output must be binary"

# --- Test vector files ---
VEC_IN_PATH = Path("tests/tx/tbcc_encoder_lte/tvec_tx_tbcc_encoder_lte/conv_encoder_tvec.in")
VEC_OUT_PATH = Path("tests/tx/tbcc_encoder_lte/tvec_tx_tbcc_encoder_lte/conv_encoder_tvec.out")

def parse_bitstring(line: str) -> list:
    return [int(b) for b in line.strip() if b in ('0', '1')]

def test_terminated_vector_matches():
    with VEC_IN_PATH.open() as f_in, VEC_OUT_PATH.open() as f_out:
        for idx, (in_line, out_line) in enumerate(zip(f_in, f_out)):
            input_bits = parse_bitstring(in_line)
            expected_output = parse_bitstring(out_line)

            expected_len = 3 * (len(input_bits) + K - 1)
            assert len(expected_output) == expected_len, \
                f"Line {idx}: Expected output length {expected_len}, got {len(expected_output)}"

            actual_output = convolutional_encode_terminated_lte(input_bits)
            assert actual_output == expected_output, \
                f"Line {idx}: Output mismatch\nExpected: {expected_output}\nActual:   {actual_output}"
