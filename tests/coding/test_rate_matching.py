import pytest
from src.tx.nr5g.polar.components.rate_matching import rate_matching

def test_rate_matching_puncturing():
    input_bits = [1, 0, 1, 1, 0, 0, 1]
    N = 7
    E = 5
    rm = 'puncturing'
    expected_output = [1, 1, 0, 0, 1]
    assert rate_matching(input_bits, rm, N, E) == expected_output

def test_rate_matching_shortening():
    input_bits = [1, 0, 1, 1, 0, 0, 1]
    N = 7
    E = 5
    rm = 'shortening'
    expected_output = [1, 0, 1, 1, 0]
    assert rate_matching(input_bits, rm, N, E) == expected_output

def test_rate_matching_repetition():
    input_bits = [1, 0, 1, 1, 0]
    N = 5
    E = 7
    rm = 'repetition'
    expected_output = [1, 0, 1, 1, 0, 1, 0]
    assert rate_matching(input_bits, rm, N, E) == expected_output

def test_rate_matching_invalid_type():
    input_bits = [1, 0, 1, 1, 0]
    N = 5
    E = 7
    rm = 'invalid'
    with pytest.raises(ValueError, match="Invalid rate matching type."):
        rate_matching(input_bits, rm, N, E)
