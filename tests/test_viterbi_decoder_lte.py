# import pytest
# from huggingface_hub import hf_hub_download
# from src.tx.encoders.tbcc_encoder_lte import convolutional_encode_tail_biting_lte
# from src.rx.decoders.tbcc.viterbi_decoder_lte import ViterbiDecoder

# REPO_ID = "furkanercan/arkastone-test-vectors"
# REPO_TYPE = "dataset"
# VEC_IN_FILE = "tx/tbcc_encoder_lte/tbcc_encoder_tvec.in"
# VEC_OUT_FILE = "tx/tbcc_encoder_lte/tbcc_encoder_tvec.out"

# def bits_from_str(bitstr):
#     return [int(b) for b in bitstr.strip()]

# def load_test_vectors_hf(limit=100):
#     vec_in_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=VEC_IN_FILE)
#     vec_out_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=VEC_OUT_FILE)

#     with open(vec_in_path, "r") as fin, open(vec_out_path, "r") as fout:
#         in_lines = [line.strip() for line in fin if line.strip()]
#         out_lines = [line.strip() for line in fout if line.strip()]
#     assert len(in_lines) == len(out_lines), "Mismatch in number of lines"
#     return in_lines[:limit], out_lines[:limit]

# @pytest.mark.parametrize("input_str, output_str", zip(*load_test_vectors_hf(limit=10)))
# def test_viterbi_with_testvecs(input_str, output_str):
#     decoder = ViterbiDecoder()
#     input_bits = bits_from_str(input_str)
#     output_bits = bits_from_str(output_str)
#     decoded = decoder.decode(output_bits, tail_biting=True)
#     assert decoded[:len(input_bits)] == input_bits
