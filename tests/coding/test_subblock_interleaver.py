import pytest
import random
from src.tx.nr5g.polar.components.subblock_interleaver import subblock_interleaver

@pytest.mark.parametrize("N", [32, 64, 128, 256, 512])
def test_subblock_interleaver(N):
    """
    Test the subblock_interleaver function to ensure it correctly interleaves the input indices
    for various values of N (from 32 to 512).
    """
    # Generate input indices
    indices = list(range(N))  # [0, 1, 2, ..., N-1]

    # Expected output based on the interleaver_map
    interleaver_map = [0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19, 
                       12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31]
    sequence_length = N // 32
    expected_output = []
    for i in interleaver_map:
        expected_output.extend(indices[i * sequence_length:(i + 1) * sequence_length])

    # Call the function
    result = subblock_interleaver(indices, N)

    # Assertions
    assert result == expected_output, f"The interleaved indices for N={N} do not match the expected output."
    assert len(result) == len(indices), f"The length of the interleaved indices for N={N} does not match the input length."

@pytest.mark.parametrize("N", [32, 64, 128, 256, 512])
def test_subblock_interleaver_random_indices(N):
    """
    Test the subblock_interleaver function with random integer indices
    for various values of N (from 32 to 512).
    """
    # Generate random input indices
    indices = random.sample(range(1000), N)  # Random integers, length N

    # Expected output based on the interleaver_map
    interleaver_map = [0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19, 
                       12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31]
    sequence_length = N // 32
    expected_output = []
    for i in interleaver_map:
        expected_output.extend(indices[i * sequence_length:(i + 1) * sequence_length])

    # Call the function
    result = subblock_interleaver(indices, N)

    # Assertions
    assert result == expected_output, f"The interleaved indices for N={N} with random input do not match the expected output."
    assert len(result) == len(indices), f"The length of the interleaved indices for N={N} with random input does not match the input length."