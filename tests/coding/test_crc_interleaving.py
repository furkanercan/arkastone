import pytest
from src.tx.nr5g.polar.components.crc_interleaving import *
# Assume your function is called crc_interleave and uses a verified interleaver pattern.

def test_output_length():
    for K in [0, 1, 10, 50, 164]:
        bits = [i % 2 for i in range(K)]
        result = crc_interleave(bits, K)
        assert len(result) == K, f"Output length mismatch for K={K}"

def test_no_duplicate_or_loss():
    K = 64
    bits = list(range(K))
    result = crc_interleave(bits, K)
    assert sorted(result) == sorted(bits), "Interleaver must preserve all bits without duplication/loss"

def test_K_0():
    result = crc_interleave([], 0)
    assert result == [], "Empty input should return empty output"

def test_K_164_identity_preservation():
    bits = list(range(164))
    result = crc_interleave(bits, 164)
    # Test only preserves elements, not ordering
    assert sorted(result) == sorted(bits), "For K=164, all elements must remain, just reordered"


def test_ref_known_interleaving_v0():
    # Reference: Fig. 12 of Egilmez et al., "Practical Aspects of Polar Code Decoding for 5G NR," 
    # IEEE Access, vol. 8, pp. 179239–179251, 2020. doi:10.1109/ACCESS.2020.3027735
    # This test uses a known interleaving example for K=36 from the paper.
    K = 36
    input_bits = [
    'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11',
    'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11',
    'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23']
    expected_output = [
    'a1', 'a4', 'a6', 'a10', 'a11', 'p0', 'a0', 'a2', 'a5', 'a7', 'p1', 'a3', 
    'a8', 'p2', 'a9', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11',
    'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23']  # Example from Fig. 12
    result = crc_interleave(input_bits, K)
    assert result == expected_output, "Mismatch with known reference result"

def test_ref_known_interleaving_v1():
    # Reference: Fig. 12 of Egilmez et al., "Practical Aspects of Polar Code Decoding for 5G NR," 
    # IEEE Access, vol. 8, pp. 179239–179251, 2020. doi:10.1109/ACCESS.2020.3027735
    # This test uses a known interleaving example for K=36 from the paper.
    K = 36
    input_bits = [0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,1,1,0,0]
    expected_output = [1,0,1,1,0,0,0,1,1,1,0,0,0,1,1,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,1,1,0,0]  # Example from Fig. 12
    result = crc_interleave(input_bits, K)
    assert result == expected_output, "Mismatch with known reference result"    

def test_ref_known_interleaving_v2():
    # Reference: Fig. 13 of Egilmez et al., "Practical Aspects of Polar Code Decoding for 5G NR," 
    # IEEE Access, vol. 8, pp. 179239–179251, 2020. doi:10.1109/ACCESS.2020.3027735
    # This test uses a known interleaving example for K=40 from the paper.
    K = 40
    input_bits = [0,1,0,1,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0]
    expected_output = [0,1,1,0,1,1,0,0,0,0,1,1,1,0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0]  # Example from Fig. 13
    result = crc_interleave(input_bits, K)
    assert result == expected_output, "Mismatch with known reference result"