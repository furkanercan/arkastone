from src.tx.nr5g.polar.components.rnti_scrambling import *

def test_rnti_scrambling():
    crc = [0] * 24
    rnti = 0xABCD
    scrambled = rnti_scramble_crc(crc, rnti)
    assert scrambled[:8] == [0]*8
    assert scrambled[8:] == [(rnti >> i) & 1 for i in reversed(range(16))]
