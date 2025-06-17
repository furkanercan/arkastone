import pytest
from src.tx.encoders.tbcc_encoder_lte import convolutional_encode_tail_biting_lte
from src.rx.decoders.tbcc.viterbi_decoder_lte import ViterbiDecoder

def load_test_vectors(in_path, out_path, limit=100):
    with open(in_path, "r") as fin, open(out_path, "r") as fout:
        input_lines = [line.strip() for line in fin if line.strip()]
        output_lines = [line.strip() for line in fout if line.strip()]
    assert len(input_lines) == len(output_lines), "Mismatch in number of lines"
    return input_lines[:limit], output_lines[:limit]

def bits_from_str(bitstr):
    return [int(b) for b in bitstr]

@pytest.mark.parametrize("input_str, output_str", zip(
    *load_test_vectors(
        "tests/tx/tbcc_encoder_lte/tvec_tx_tbcc_encoder_lte/tbcc_encoder_tvec.in",
        "tests/tx/tbcc_encoder_lte/tvec_tx_tbcc_encoder_lte/tbcc_encoder_tvec.out",
        limit=10
    )
))
def test_viterbi_with_testvecs(input_str, output_str):
    decoder = ViterbiDecoder()
    input_bits = bits_from_str(input_str)
    output_bits = bits_from_str(output_str)
    decoded = decoder.decode(output_bits, tail_biting=True)

    assert decoded[:len(input_bits)] == input_bits