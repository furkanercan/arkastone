import math

def subchannel_allocation(input_bits, info_indices, pc_indices, output_bits):
    """
    Map the input bits to the output bits based on the provided information indices.
    The output_bits list must be passed as an argument and initialized to zeros with length N.

    Args:
        input_bits (list): The list of input bits of length K.
        info_indices (list): The list of indices where input bits should be placed.
        output_bits (list): The list of output bits of length N, initialized to zeros.

    Returns:
        list: The updated output_bits with input_bits placed at the info_indices.
    """
    for i, index in enumerate(info_indices):
        output_bits[index] = input_bits[i]
    
    for i, index in enumerate(pc_indices):
        output_bits[index] = pc_indices[i]
    