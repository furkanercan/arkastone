def rate_matching(input_bits, rm, N, E):
    """
    Perform rate matching on the input bits.

    Parameters:
    - input_bits: List of input bits.
    - rm: Rate matching type ('puncturing', 'shortening', 'repetition').
    - N: Length of the input bits.
    - E: Length of the output bits.

    Returns:
    - List of output bits after rate matching.
    """
    if rm == 'puncturing':
        # Remove the first N-E bits
        output_bits = input_bits[N-E:]
    elif rm == 'shortening':
        # Remove the last N-E bits
        output_bits = input_bits[:E]
    elif rm == 'repetition':
        # Append the first E-N bits to the end
        output_bits = input_bits + input_bits[:E-N]
    else:
        raise ValueError("Invalid rate matching type. Use 'puncturing', 'shortening', or 'repetition'.")

    return output_bits