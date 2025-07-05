import pytest
import numpy as np
from huggingface_hub import hf_hub_download
from src.tx.nr5g.polar.components.polar_encoder_core import polar_encode

REPO_ID = "furkanercan/arkastone-test-vectors"
REPO_TYPE = "dataset"

@pytest.fixture(scope="module")
def hf_test_vectors():
    gmat_file = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="coding/polar_gmat_64x128/Gmat.csv")
    uncoded_file = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="coding/polar_gmat_64x128/uncoded_data.csv")
    encoded_file = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="coding/polar_gmat_64x128/encoded_data.csv")
    return gmat_file, uncoded_file, encoded_file

def test_polar_encode(hf_test_vectors):
    gmat_file, uncoded_file, encoded_file = hf_test_vectors

    Gmat_kxN = np.loadtxt(gmat_file, delimiter=',', dtype=int)
    uncoded_data = np.loadtxt(uncoded_file, delimiter=',', dtype=int)
    encoded_data = np.loadtxt(encoded_file, delimiter=',', dtype=int)

    assert uncoded_data.shape[0] == encoded_data.shape[0], "Mismatch in number of test cases."

    for i in range(uncoded_data.shape[0]):
        input_bits = uncoded_data[i]
        expected = encoded_data[i]
        actual = polar_encode(input_bits, Gmat_kxN)

        assert np.array_equal(actual, expected), (
            f"Test case {i} failed:\nInput:    {input_bits}\nExpected: {expected}\nGot:      {actual}"
        )
