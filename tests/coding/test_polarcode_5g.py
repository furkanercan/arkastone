import pytest
from src.coding.polar.nr5g.polarcode_5g import PolarCodeChannelConfig

# ✅ Valid config: typical PUSCH values (A=56, G=864)
def test_valid_pusch():
    config = PolarCodeChannelConfig(A=56, G=864, channel_type="PUSCH")
    assert config.validate()

# ✅ Valid config: PUSCH with segmentation enabled (A=1200, G=16384)
def test_valid_pusch_with_segmentation():
    config = PolarCodeChannelConfig(A=1200, G=16384, channel_type="PUSCH")
    assert config.segmentation
    assert config.validate()

# ✅ Valid config: PC bits active due to low A and relatively high G
def test_valid_pusch_low_A_high_G_pc_bits():
    config = PolarCodeChannelConfig(A=15, G=220, channel_type="PUSCH")
    assert config.pc_bits == 3
    assert config.pc_row_weight == 1

# ✅ Valid config: PDCCH with A < 12 (should emit a warning, not fail)
def test_valid_pdcch_warning():
    config = PolarCodeChannelConfig(A=8, G=300, channel_type="PDCCH")
    assert config.validate()

# ❌ Invalid PBCH config: A must be exactly 32
def test_invalid_pbch_A():
    with pytest.raises(ValueError, match="A must be 32"):
        PolarCodeChannelConfig(A=31, G=864, channel_type="PBCH")

# ❌ Invalid PUCCH config: G exceeds 16384 even with segmentation
def test_invalid_ul_G_too_high():
    with pytest.raises(ValueError, match="exceeds 16384"):
        PolarCodeChannelConfig(A=1400, G=17000, channel_type="PUCCH")

# ❌ Invalid PDCCH config: G is below allowed minimum
def test_invalid_dl_G_too_low():
    with pytest.raises(ValueError, match="must be between 25"):
        PolarCodeChannelConfig(A=50, G=20, channel_type="PDCCH")

# ❌ Invalid config: channel type is unsupported
def test_invalid_channel_type():
    with pytest.raises(ValueError, match="Unsupported channel type"):
        PolarCodeChannelConfig(A=50, G=200, channel_type="XYZ")
