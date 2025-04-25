import pytest
from src.utils.bit_symbol_utils import BitSymbolUtils

def test_pack_bits_to_symbols():
    bitstream = [1, 0, 1, 1, 0, 0, 1, 0]  # Binary: 10110010 = 0xB2 = 178
    symbols = BitSymbolUtils.pack_bits_to_symbols(bitstream, symbol_size=8)
    assert symbols == [178], f"Expected [178], got {symbols}"

def test_unpack_symbols_to_bits():
    symbols = [178]  # 0xB2 = 10110010 in binary
    bitstream = BitSymbolUtils.unpack_symbols_to_bits(symbols, symbol_size=8)
    assert bitstream == [1, 0, 1, 1, 0, 0, 1, 0], f"Expected [1, 0, 1, 1, 0, 0, 1, 0], got {bitstream}"

def test_pack_unpack_consistency():
    original_bitstream = [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    symbols = BitSymbolUtils.pack_bits_to_symbols(original_bitstream, symbol_size=8)
    unpacked_bitstream = BitSymbolUtils.unpack_symbols_to_bits(symbols, symbol_size=8)
    assert unpacked_bitstream == original_bitstream, f"Mismatch: {unpacked_bitstream} vs {original_bitstream}"
