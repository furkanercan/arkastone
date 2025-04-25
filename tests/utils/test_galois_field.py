import pytest
from src.utils.galois_field import GF

@pytest.fixture
def gf():
    return GF(m=8, primitive_poly=0x11D)  # GF(2^8) with standard primitive polynomial

def test_addition(gf):
    assert gf.add(5, 7) == 5 ^ 7  # XOR operation
    assert gf.add(0, 10) == 10
    assert gf.add(255, 255) == 0

def test_multiplication(gf):
    assert gf.mul(3, 7) == gf.exp_table[(gf.log_table[3] + gf.log_table[7]) % 255]
    assert gf.mul(1, 200) == 200  # Identity property
    assert gf.mul(0, 55) == 0  # Zero property

def test_division(gf):
    assert gf.div(6, 3) == gf.mul(6, gf.inv(3))
    assert gf.div(200, 1) == 200  # Identity property
    with pytest.raises(ZeroDivisionError):
        gf.div(100, 0)  # Should raise error

def test_inverse(gf):
    assert gf.mul(50, gf.inv(50)) == 1  # Multiplicative inverse check
    with pytest.raises(ZeroDivisionError):
        gf.inv(0)  # Zero has no inverse
