import csv
import pytest
from src.tx.nr5g.polar.polar_nr5g_wrapper import PolarNR5GWrapper

# Mapping of rate matching codes in the CSV to their full names
RATE_MATCHING_MAP = {
    "S": "shortening",
    "P": "puncturing",
    "R": "repetition"
}

@pytest.mark.parametrize("csv_file", [
    "tests/coding/resources/nr5g_polar_test_data_valerio.csv"
])
def test_resource_based(csv_file):
    """
    Test PolarNR5GWrapper against a resource file containing expected values for
    N, rate matching scheme, and segmentation flag.
    """
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for line_number, row in enumerate(reader, start=2):  # Start at 2 to account for the header row
            # Read inputs from the CSV
            G = int(row['G'])
            A = int(row['A'])
            channel_type = row['Channel']
            expected_N = int(row['N'])
            expected_rate_matching = RATE_MATCHING_MAP[row['RateM']]  # Map the rate matching code
            expected_segmentation = bool(int(row['Seg']))

            # Initialize the wrapper
            wrapper = PolarNR5GWrapper(A, G, channel_type)

            # Assertions for N
            assert wrapper.N == expected_N, (
                f"Line {line_number}: N mismatch for G={G}, A={A}, Channel={channel_type}: "
                f"expected {expected_N}, got {wrapper.N}"
            )

            # Assertions for rate matching scheme
            assert wrapper.rm == expected_rate_matching, (
                f"Line {line_number}: Rate matching scheme mismatch for G={G}, A={A}, Channel={channel_type}: "
                f"expected {expected_rate_matching}, got {wrapper.rm}"
            )

            # Assertions for segmentation flag
            assert wrapper.segmentation == expected_segmentation, (
                f"Line {line_number}: Segmentation flag mismatch for G={G}, A={A}, Channel={channel_type}: "
                f"expected {expected_segmentation}, got {wrapper.segmentation}"
            )