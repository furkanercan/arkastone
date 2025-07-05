import csv
import pytest
from huggingface_hub import hf_hub_download
from src.tx.nr5g.polar.polar_nr5g_wrapper import PolarNR5GWrapper

REPO_ID = "furkanercan/arkastone-test-vectors"
REPO_TYPE = "dataset"
CSV_FILENAME = "coding/polar_nr5g_wrapper/polar_nr5g_wrapper_test_data_valerio.csv"

RATE_MATCHING_MAP = {
    "S": "shortening",
    "P": "puncturing",
    "R": "repetition"
}

@pytest.fixture(scope="module")
def csv_path():
    return hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=CSV_FILENAME)

def test_resource_based(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for line_number, row in enumerate(reader, start=2):  # Start at 2 to account for the header
            G = int(row['G'])
            A = int(row['A'])
            channel_type = row['Channel']
            expected_N = int(row['N'])
            expected_rate_matching = RATE_MATCHING_MAP[row['RateM']]
            expected_segmentation = bool(int(row['Seg']))

            wrapper = PolarNR5GWrapper(A, G, channel_type)

            assert wrapper.N == expected_N, (
                f"Line {line_number}: N mismatch for G={G}, A={A}, Channel={channel_type}: "
                f"expected {expected_N}, got {wrapper.N}"
            )

            assert wrapper.rm == expected_rate_matching, (
                f"Line {line_number}: Rate matching mismatch for G={G}, A={A}, Channel={channel_type}: "
                f"expected {expected_rate_matching}, got {wrapper.rm}"
            )

            assert wrapper.segmentation == expected_segmentation, (
                f"Line {line_number}: Segmentation flag mismatch for G={G}, A={A}, Channel={channel_type}: "
                f"expected {expected_segmentation}, got {wrapper.segmentation}"
            )
