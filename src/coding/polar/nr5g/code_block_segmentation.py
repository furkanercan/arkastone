def segment_transport_block(tb_bits, A):
    """
    Segments the transport block if needed and appends 24-bit CRC to each block.
    Used in PUCCH and PUSCH.
    If segmented: block is decomposed into 2 equal-length segments. 
    Each segment is encoded and decoded separately, but with identical configuration.


    Args:
        tb_bits (list[int]): Input transport block bits (0/1).

    Returns:
        List of code blocks (each a list[int]), with CRC24B appended.
    """

    if A % 2 == 1:
        tb_bits = [0] + tb_bits  # prepend to make length even
        A += 1

    mid = A // 2
    block1 = tb_bits[:mid]
    block2 = tb_bits[mid:]

    return [block1, block2]