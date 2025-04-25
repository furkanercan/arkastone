class BitSymbolUtils:
    """
    Utility functions for packing/unpacking bits to symbols and vice versa.
    """
    @staticmethod
    def pack_bits_to_symbols(bitstream, symbol_size=8):
        """Converts a bitstream into Reed-Solomon symbols."""
        assert len(bitstream) % symbol_size == 0, "Bitstream length must be a multiple of symbol_size"
        symbols = [int("".join(map(str, bitstream[i:i + symbol_size])), 2)
                   for i in range(0, len(bitstream), symbol_size)]
        return symbols

    @staticmethod
    def unpack_symbols_to_bits(symbols, symbol_size=8):
        """Converts Reed-Solomon symbols back into a bitstream."""
        bitstream = [int(b) for symbol in symbols for b in format(symbol, f'0{symbol_size}b')]
        return bitstream