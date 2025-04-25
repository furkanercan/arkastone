def concatenate_code_blocks(blocks: list[list[int]], G: int) -> list[int]:
    """
    Concatenates segmented code blocks into a transport block.
    Pads with a zero if G is odd.

    Args:
        blocks (list of list of int): Decoded blocks, CRC removed.
        G (int): Final encoded length (used to check if a 0-padding is needed).

    Returns:
        list[int]: Concatenated transport block bits.
    """
    if not blocks:
        return []

    if len(blocks) == 1:
        tb = blocks[0]
    elif len(blocks) == 2:
        tb = blocks[0] + blocks[1]
        if G % 2 == 1: 
            tb.append(0) # pad only when G is odd and C=2
    else:
        raise ValueError("Only up to 2 code blocks supported.")

    return tb
