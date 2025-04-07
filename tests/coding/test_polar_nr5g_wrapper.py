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
    (550, 16384, 'PUCCH', True),  # condition 1 fails, condition 2 passes
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
    (550, 16384, 'PUSCH', True),  # condition 1 fails, condition 2 passes
    #PDCCH
    (1, 25, 'PDCCH', False),
    (1, 8192, 'PDCCH', False),
    (140, 250, 'PDCCH', False),
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

@pytest.mark.parametrize("expected_N, expected_indices", [
    (64, [63, 62, 61, 59, 55, 47, 31, 60, 58, 57, 54, 53, 46, 51, 45, 30, 43, 
          29, 39, 27, 56, 23, 52, 15, 50, 44, 49, 42, 28, 41, 38, 22, 25, 37, 
          26, 35, 21, 14, 48, 13, 19, 40, 11, 7, 36, 24, 34, 20, 33, 12, 18, 
          10, 17, 6, 9, 5, 3, 32, 16, 8, 4, 2, 1, 0]),
])
def test_get_reliability_indices_n64(expected_N, expected_indices):
    wrapper = PolarNR5GWrapper(25, 50, 'PUCCH')  # Example initialization
    assert wrapper.reliability_indices == expected_indices
    assert wrapper.N == expected_N

@pytest.mark.parametrize("expected_N, expected_indices", [
    (512, [511, 510, 509, 507, 503, 495, 508, 479, 506, 505, 447, 501, 494, 502, 
    499, 493, 383, 478, 491, 477, 255, 504, 487, 475, 446, 500, 471, 445, 498, 
    382, 443, 492, 497, 381, 463, 490, 439, 476, 486, 489, 431, 379, 254, 474, 
    473, 485, 415, 483, 470, 444, 375, 253, 367, 247, 469, 441, 442, 462, 251, 
    438, 467, 351, 496, 461, 380, 437, 459, 378, 239, 488, 430, 484, 319, 435, 
    377, 455, 472, 223, 414, 427, 482, 373, 252, 429, 468, 366, 413, 481, 371, 
    250, 466, 423, 374, 440, 365, 411, 249, 460, 350, 246, 465, 436, 407, 191, 
    127, 363, 458, 245, 349, 434, 399, 457, 359, 238, 428, 376, 318, 454, 243, 
    347, 433, 237, 453, 426, 222, 317, 372, 343, 412, 235, 451, 425, 422, 370, 
    221, 315, 480, 335, 364, 190, 369, 248, 231, 410, 421, 311, 219, 409, 362, 
    464, 406, 419, 348, 215, 361, 189, 244, 303, 405, 358, 456, 346, 398, 242, 
    126, 236, 187, 357, 432, 207, 403, 397, 452, 345, 241, 316, 342, 125, 234, 
    183, 287, 355, 395, 424, 314, 220, 341, 123, 175, 233, 334, 450, 313, 391, 
    230, 368, 218, 339, 119, 333, 310, 420, 159, 229, 408, 217, 449, 188, 309, 
    214, 331, 111, 360, 302, 418, 227, 404, 186, 213, 417, 301, 307, 356, 402, 
    327, 95, 206, 240, 344, 396, 185, 401, 211, 354, 299, 286, 182, 205, 124, 232, 
    285, 295, 181, 394, 340, 63, 203, 353, 448, 122, 283, 393, 174, 390, 312, 338, 
    228, 179, 199, 121, 173, 389, 332, 118, 337, 158, 279, 271, 416, 216, 308, 387, 
    226, 330, 171, 212, 117, 110, 329, 157, 306, 326, 225, 167, 115, 184, 109, 300, 
    305, 210, 155, 325, 352, 400, 298, 204, 94, 284, 209, 151, 180, 107, 297, 392, 
    323, 202, 93, 294, 178, 103, 143, 282, 62, 336, 201, 120, 172, 198, 91, 388, 293, 
    177, 278, 281, 61, 170, 116, 197, 87, 156, 277, 114, 169, 59, 291, 275, 270, 195, 
    166, 224, 108, 269, 79, 154, 113, 328, 55, 106, 165, 153, 150, 386, 208, 324, 385, 
    267, 47, 92, 163, 296, 304, 105, 102, 149, 263, 322, 292, 90, 200, 31, 321, 142, 
    176, 147, 101, 141, 196, 290, 89, 280, 60, 86, 99, 139, 168, 58, 276, 85, 194, 
    289, 78, 135, 112, 57, 83, 54, 274, 268, 164, 77, 152, 193, 53, 162, 104, 273, 
    266, 75, 46, 148, 51, 100, 45, 161, 265, 262, 71, 146, 30, 140, 88, 98, 43, 29, 
    261, 145, 138, 84, 259, 39, 97, 27, 56, 82, 137, 76, 384, 134, 23, 52, 133, 320, 
    15, 73, 50, 81, 131, 44, 70, 192, 288, 160, 272, 74, 49, 42, 69, 28, 144, 41, 
    67, 96, 38, 264, 260, 136, 22, 25, 37, 80, 26, 258, 35, 132, 21, 257, 72, 14, 
    48, 13, 19, 130, 68, 40, 11, 66, 129, 7, 36, 24, 34, 256, 20, 65, 33, 12, 128, 
    18, 10, 17, 6, 9, 64, 5, 3, 32, 16, 8, 4, 2, 1, 0 ]),
])
def test_get_reliability_indices_n512(expected_N, expected_indices):
    wrapper = PolarNR5GWrapper(32, 864, 'PBCH')  # Example initialization
    assert wrapper.reliability_indices == expected_indices
    assert wrapper.N == expected_N

@pytest.mark.parametrize("A, G, channel_type", [
    (11, 100, 'PUCCH'),  # A is below A_min
    (1707, 100, 'PUCCH'),  # A is above A_max
    (100, 30, 'PUCCH'),  # G is below G_min
    (100, 16385, 'PUCCH'),  # G is above G_max
    (0, 50, 'PDCCH'),  # A is below A_min for PDCCH
    (150, 50, 'PDCCH'),  # A is above A_max for PDCCH
    (100, 20, 'PDCCH'),  # G is below G_min for PDCCH
    (100, 9000, 'PDCCH'),  # G is above G_max for PDCCH
    (31, 800, 'PBCH'),  # A is below A_min for PBCH
    (33, 800, 'PBCH'),  # A is above A_max for PBCH
    (32, 863, 'PBCH'),  # G is below G_min for PBCH
    (32, 865, 'PBCH'),  # G is above G_max for PBCH
    (100, 100, 'PUCCH'),  # A is equal to G
    (900, 864, 'PBCH'),  # A is greater than G
])
def test_validate_invalid(A, G, channel_type):
    """
    Test that the PolarNR5GWrapper class raises a ValueError during initialization
    for invalid A or G values, or when A is not smaller than G.
    """
    with pytest.raises(ValueError) as excinfo:
        PolarNR5GWrapper(A, G, channel_type)  # Validation happens during initialization
    # Ensure the error message includes the channel type
    assert channel_type in str(excinfo.value)

@pytest.mark.parametrize("A, G, channel_type", [
    (12, 31, 'PUCCH'),  # Valid A and G for PUCCH
    (1706, 8192, 'PUCCH'),  # Valid A and G for PUCCH
    (1, 25, 'PDCCH'),  # Valid A and G for PDCCH
    (140, 8192, 'PDCCH'),  # Valid A and G for PDCCH
    (32, 864, 'PBCH'),  # Valid A and G for PBCH
])
def test_validate_valid(A, G, channel_type):
    """
    Test that the validate function does not raise an error for valid A and G values.
    """
    wrapper = PolarNR5GWrapper(A, G, channel_type)
    try:
        wrapper.validate()
    except ValueError:
        pytest.fail(f"validate() raised ValueError unexpectedly for channel type '{channel_type}'")
