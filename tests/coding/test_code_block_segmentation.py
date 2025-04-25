import pytest
from src.tx.nr5g.polar.components.code_block_segmentation import segment_transport_block

def test_segmentation_even_length():
    tb = [1, 0, 0, 1, 1, 1]  # A = 6
    result = segment_transport_block(tb, len(tb))
    assert len(result) == 2

    expected_block1 = [1, 0, 0]
    expected_block2 = [1, 1, 1]

    assert result[0] == expected_block1
    assert result[1] == expected_block2

def test_segmentation_odd_length():
    tb = [1, 1, 0, 1, 0]  # A = 5 → becomes [0, 1, 1, 0, 1, 0]
    result = segment_transport_block(tb, len(tb))
    assert len(result) == 2

    expected_tb = [0] + tb  # padded
    mid = len(expected_tb) // 2
    expected_block1 = expected_tb[:mid]  # [0, 1, 1]
    expected_block2 = expected_tb[mid:]  # [0, 1, 0]

    assert result[0] == expected_block1
    assert result[1] == expected_block2
    assert len(result[0]) == len(result[1])
