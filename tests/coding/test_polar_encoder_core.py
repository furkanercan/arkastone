import pytest
import numpy as np
from src.tx.nr5g.polar.components.polar_encoder_core import polar_encode

@pytest.mark.parametrize("gmat_file, uncoded_file, encoded_file", [
    ("tests/coding/resources/polar_encoder_core/gmat_64x128/Gmat.csv", 
     "tests/coding/resources/polar_encoder_core/gmat_64x128/uncoded_data.csv", 
     "tests/coding/resources/polar_encoder_core/gmat_64x128/encoded_data.csv"),
])
def test_polar_encode(gmat_file, uncoded_file, encoded_file):
    """
    Test the polar_encode function using the provided Gmat, uncoded_data, and encoded_data files.
    """
    # Load the generator matrix (Gmat_kxN)
    Gmat_kxN = np.loadtxt(gmat_file, delimiter=',', dtype=int)

    # Load the uncoded data (input bits) and encoded data (expected output)
    uncoded_data = np.loadtxt(uncoded_file, delimiter=',', dtype=int)
    encoded_data = np.loadtxt(encoded_file, delimiter=',', dtype=int)

    # Ensure the number of cases matches between uncoded and encoded data
    assert uncoded_data.shape[0] == encoded_data.shape[0], "Mismatch in number of test cases between uncoded and encoded data."

    # Iterate through all test cases
    for i in range(uncoded_data.shape[0]):
        input_bits = uncoded_data[i]
        expected_encoded_bits = encoded_data[i]

        # Perform polar encoding
        generated_encoded_bits = polar_encode(input_bits, Gmat_kxN)

        # Compare the generated encoded bits with the expected encoded bits
        assert np.array_equal(generated_encoded_bits, expected_encoded_bits), (
            f"Mismatch in encoded bits for test case {i}:\n"
            f"Input bits: {input_bits}\n"
            f"Expected: {expected_encoded_bits}\n"
            f"Generated: {generated_encoded_bits}"
        )