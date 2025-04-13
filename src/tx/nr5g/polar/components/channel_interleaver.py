def channel_interleaver(input_bits, channel_interleaved_index):
    """
    Maps the input_bits to output_bits using the channel_interleaved_index as the guide.

    Args:
        input_bits (list or array): The input bits of length E.
        channel_interleaved_index (list or array): The interleaving index of length E.

    Returns:
        list: The output bits of length E after interleaving.
    """
    if len(input_bits) != len(channel_interleaved_index):
        raise ValueError("Length of input_bits and channel_interleaved_index must be the same.")

    E = len(input_bits)
    output_bits = [0] * E

    for i in range(E):
        output_bits[channel_interleaved_index[i]] = input_bits[i]

    return output_bits