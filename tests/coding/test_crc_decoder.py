import numpy as np
import pytest
from src.configs.config_crc import CRCConfig
from src.coding.crc.crc_encoder import CRCEncoder
from src.coding.crc.crc_decoder import CRCDecoder

@pytest.mark.parametrize("crc_length", [4, 6, 8, 11, 12, 16, 24, 32])
@pytest.mark.parametrize("info_len", [4, 8, 16, 25, 76, 31, 1024, 1111])
def test_crc_decoder_various_lengths(crc_length, info_len):
    config = CRCConfig(
        enable=True,
        name=f"CRC{crc_length}",
        length=crc_length,
        preload_val=0,
        mode='generic'  # Assuming internal logic handles poly from length/name
    )

    encoder = CRCEncoder(config)
    decoder = CRCDecoder(config)

    rng = np.random.default_rng(seed=info_len + crc_length)
    info_bits = rng.integers(0, 2, size=info_len)

    codeword = encoder.encode_and_append(info_bits)
    assert decoder.crc_decode(codeword, len_k=info_len)

    corrupted = codeword.copy()
    corrupted[-1] ^= 1
    assert not decoder.crc_decode(corrupted, len_k=info_len)
