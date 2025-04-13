import pytest
from src.tx.nr5g.polar.components.channel_interleaver import channel_interleaver

def test_channel_interleaver_basic():
    input_bits = [1, 0, 1, 1, 0]
    channel_interleaved_index = [4, 2, 0, 3, 1]
    expected_output = [1, 0, 0, 1, 1]
    assert channel_interleaver(input_bits, channel_interleaved_index) == expected_output

def test_channel_interleaver_invalid_length():
    input_bits = [1, 0, 1]
    channel_interleaved_index = [2, 0, 1, 3]
    with pytest.raises(ValueError, match="Length of input_bits and channel_interleaved_index must be the same."):
        channel_interleaver(input_bits, channel_interleaved_index)

def test_channel_interleaver_large_input():
    input_bits = list(range(10))
    channel_interleaved_index = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    expected_output = list(reversed(input_bits))
    assert channel_interleaver(input_bits, channel_interleaved_index) == expected_output
