import pytest
from src.coding.polar.nr5g.polar_nr5g_wrapper import PolarNR5GWrapper


@pytest.mark.parametrize("A, G, channel_type, expected_segmentation", [
    #PUCCH
    (1020, 1050, 'PUCCH', True),  # condition 1 fails, condition 2 fails
    (1013, 1036, 'PUCCH', True),  # condition 1 passes, condition 2 fails
    (1013, 1088, 'PUCCH', True),  # condition 1 passes, condition 2 passes
    (1066, 1088, 'PUCCH', True),  # condition 1 passes, condition 2 passes
    (1012, 1036, 'PUCCH', False), # condition 1 fails, condition 2 fails
    (1012, 1088, 'PUCCH', True),  # condition 1 fails, condition 2 passes
    (1012, 1087, 'PUCCH', False), # condition 1 fails, condition 2 fails
    (1050, 2503, 'PUCCH', True),  # condition 1 fails, condition 2 passes
    (359, 1088, 'PUCCH', False),  # condition 1 fails, condition 2 fails
    (360, 1088, 'PUCCH', True),   # condition 1 fails, condition 2 passes
    (1706, 2503, 'PUCCH', True),  # condition 1 fails, condition 2 passes
    (550, 16385, 'PUCCH', True),  # condition 1 fails, condition 2 passes
    #PUSCH
    (1020, 1050, 'PUSCH', True),  # condition 1 fails, condition 2 fails
    (1013, 1036, 'PUSCH', True),  # condition 1 passes, condition 2 fails
    (1013, 1088, 'PUSCH', True),  # condition 1 passes, condition 2 passes
    (1066, 1088, 'PUSCH', True),  # condition 1 passes, condition 2 passes
    (1012, 1036, 'PUSCH', False), # condition 1 fails, condition 2 fails
    (1012, 1088, 'PUSCH', True),  # condition 1 fails, condition 2 passes
    (1012, 1087, 'PUSCH', False), # condition 1 fails, condition 2 fails
    (1050, 2503, 'PUSCH', True),  # condition 1 fails, condition 2 passes
    (359, 1088, 'PUSCH', False),  # condition 1 fails, condition 2 fails
    (360, 1088, 'PUSCH', True),   # condition 1 fails, condition 2 passes
    (1706, 2503, 'PUSCH', True),  # condition 1 fails, condition 2 passes
    (550, 16385, 'PUSCH', True),  # condition 1 fails, condition 2 passes
    #PDCCH
    (1, 25, 'PDCCH', False),
    (1, 8192, 'PDCCH', False),
    (140, 25, 'PDCCH', False),
    (140, 8192, 'PDCCH', False),
    (86, 2487, 'PDCCH', False),
    (33, 1333, 'PDCCH', False),
    (8, 64, 'PDCCH', False),
    #PBCH
    (32, 864, 'PBCH', False),
    # TODO: Add more test cases to cover violations of the conditions
])
def test_segmentation_flag(A, G, channel_type, expected_segmentation):
    wrapper = PolarNR5GWrapper(A, G, channel_type)
    assert wrapper.segmentation == expected_segmentation

@pytest.mark.parametrize("A, G, channel_type, expected_crc_name", [
    (25, 100, 'PUCCH', "CRC11"),
    (1000, 3000, 'PUCCH', "CRC11"),
    (15, 50, 'PUSCH', "CRC6"),
    (12, 2000, 'PUCCH', "CRC6"),
    (10, 50, 'PDCCH', "CRC24"),
    (32, 864, 'PBCH', "CRC24"),
])
def test_coding_parameters(A, G, channel_type, expected_crc_name):
    wrapper = PolarNR5GWrapper(A, G, channel_type)
    assert wrapper.crc.name == expected_crc_name

@pytest.mark.parametrize("A, G, channel_type, expected_rm", [
    (25, 50, 'PUCCH', 'shortening'),
    (15, 100, 'PUSCH', 'puncturing'),
    (10, 200, 'PDCCH', 'puncturing'),
    (32, 864, 'PBCH', 'repetition'),
])
def test_rate_matching_scheme(A, G, channel_type, expected_rm):
    wrapper = PolarNR5GWrapper(A, G, channel_type)
    assert wrapper.rm == expected_rm

# def test_encode_pucch():
#     wrapper = PolarNR5GWrapper(25, 100, 'PUCCH')
#     input_bits = [1, 0, 1, 0, 1]
#     encoded_bits = wrapper.encode(input_bits)
#     assert isinstance(encoded_bits, list)  # Add more specific checks as needed

# def test_decode_pucch():
#     wrapper = PolarNR5GWrapper(25, 100, 'PUCCH')
#     received_bits = [1, 0, 1, 0, 1]
#     decoded_bits = wrapper.decode(received_bits)
#     assert isinstance(decoded_bits, list)  # Add more specific checks as needed
