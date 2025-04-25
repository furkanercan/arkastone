
def subblock_interleaver(input_sequence, N):
    """
    Apply the interleaver to the given input sequence.
    """
    interleaver_map = [0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19, 
                        12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31]
    sequence_length = N // 32
    interleaved_indices = []
    for i in interleaver_map:
        interleaved_indices.extend(input_sequence[i * sequence_length:(i + 1) * sequence_length])
    return interleaved_indices
