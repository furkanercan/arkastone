import numpy as np

def polar_encode(input_bits, Gmat_kxN):
    """
    Perform polar encoding using vector-matrix multiplication.

    Parameters:
    input_bits (numpy.ndarray): Input bit vector of size k.
    Gmat_kxN (numpy.ndarray): Generator matrix of size k x N.

    Returns:
    numpy.ndarray: Encoded bit vector of size N.
    """
    # Ensure input_bits is a numpy array
    input_bits = np.array(input_bits, dtype=int)

    # Perform vector-matrix multiplication in GF(2)
    encoded_bits = np.mod(np.dot(input_bits, Gmat_kxN), 2)

    return encoded_bits