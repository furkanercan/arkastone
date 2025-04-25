import pytest
from src.tx.nr5g.polar.components.code_block_concatenation import concatenate_code_blocks

def test_single_block_no_padding_g0():
    block = [[1, 0, 1, 1]]
    G = 4
    result = concatenate_code_blocks(block, G)
    assert result == [1, 0, 1, 1]

def test_single_block_no_padding_g1():
    block = [[1, 0, 1, 1, 1, 0, 1]]
    G = 7
    result = concatenate_code_blocks(block, G)
    assert result == [1, 0, 1, 1, 1, 0, 1]

def test_two_blocks_no_padding():
    blocks = [[1, 1], [0, 0]]
    G = 4
    result = concatenate_code_blocks(blocks, G)
    assert result == [1, 1, 0, 0]

def test_two_blocks_with_padding():
    blocks = [[1, 1], [0, 0]]
    G = 5  # odd → one 0 should be appended
    result = concatenate_code_blocks(blocks, G)
    assert result == [1, 1, 0, 0, 0]

def test_empty_input():
    blocks = []
    G = 0
    result = concatenate_code_blocks(blocks, G)
    assert result == []

def test_invalid_block_count():
    blocks = [[1], [0], [1]]
    G = 6
    with pytest.raises(ValueError):
        concatenate_code_blocks(blocks, G)