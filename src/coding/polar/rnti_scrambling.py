def rnti_scramble_crc(crc_bits: list[int], rnti: int) -> list[int]:
    """
    XORs the last 16 bits of the 24-bit CRC with the 16-bit RNTI value.

    Args:
        crc_bits (list[int]): 24-bit CRC as list of ints (0 or 1)
        rnti (int): 16-bit RNTI (integer)

    Returns:
        list[int]: Scrambled 24-bit CRC
    """
    if len(crc_bits) != 24:
        raise ValueError("CRC must be 24 bits long.")

    rnti_bits = [(rnti >> i) & 1 for i in reversed(range(16))]  # MSB-first
    scrambled_crc = crc_bits[:8] + [b ^ r for b, r in zip(crc_bits[8:], rnti_bits)]

    return scrambled_crc
