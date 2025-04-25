import pytest
from src.tx.encoders.encoder_rs import ReedSolomonEncoder

@pytest.fixture
def rs_encoder():
    return ReedSolomonEncoder(n=255, k=223)

def test_rs_encoding_length(rs_encoder):
    data = [i % 256 for i in range(223)]  # Example data symbols
    codeword = rs_encoder.encode(data)
    assert len(codeword) == 255, f"Expected length 255, got {len(codeword)}"

def test_rs_encoding_structure(rs_encoder):
    data = [i for i in range(223)]  # Example data symbols
    codeword = rs_encoder.encode(data)
    assert codeword[:223] == data, "Encoded message should start with input data"
    assert any(codeword[223:]), "Parity symbols should not be all zero"

def test_rs_encoding_different_inputs(rs_encoder):
    data1 = [i for i in range(223)]
    data2 = [i + 1 for i in range(223)]
    codeword1 = rs_encoder.encode(data1)
    codeword2 = rs_encoder.encode(data2)
    assert codeword1 != codeword2, "Different inputs should produce different codewords"

def test_rs_encoding_all_zero_input(rs_encoder):
    data = [0] * 223  # All zeros input
    codeword = rs_encoder.encode(data)
    assert all(symbol == 0 for symbol in codeword), f"Expected all-zero codeword, got {codeword}"
    # assert codeword[:223] == data, "Encoded message should retain input data"
    # assert any(codeword[223:]), "Parity symbols should not be all zero"
    
def test_rs_encoding_all_max_value(rs_encoder):
    data = [255] * 223  # All max values (0xFF)
    codeword = rs_encoder.encode(data)
    assert codeword[:223] == data, "Encoded message should retain input data"
    assert any(codeword[223:]), "Parity symbols should not be all zero"

def test_rs_encoding_single_value_change(rs_encoder):
    data1 = [100] * 223
    data2 = [100] * 222 + [101]  # One value changed
    codeword1 = rs_encoder.encode(data1)
    codeword2 = rs_encoder.encode(data2)
    assert codeword1 != codeword2, "Changing one symbol should change the codeword"