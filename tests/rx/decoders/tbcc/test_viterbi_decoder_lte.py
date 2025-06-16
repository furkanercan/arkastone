# import pytest
# from src.tx.encoders.tbcc_encoder_lte import convolutional_encode_tail_biting_lte
# from src.rx.decoders.tbcc.viterbi_decoder_lte import ViterbiDecoder

# @pytest.mark.parametrize("input_bits", [
#     [0, 0, 0, 0],
#     [1, 0, 1, 1],
#     [1, 1, 1, 1],
#     [0, 1, 0, 1, 0, 1],
#     [1, 0, 1, 1, 0, 0, 1, 1],
# ])
# def test_viterbi_decoder_with_lte_encoder(input_bits):
#     decoder = ViterbiDecoder()
#     encoded = convolutional_encode_tail_biting_lte(input_bits)
#     decoded = decoder.decode(encoded)
#     # Compare only the first N bits (since decoder may output more)
#     assert decoded[:len(input_bits)] == input_bits