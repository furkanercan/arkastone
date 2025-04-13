import pytest
import numpy as np
from src.tx.nr5g.polar.polar_nr5g_wrapper import PolarNR5GWrapper


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
def test_coding_parameters_crc(A, G, channel_type, expected_crc_name):
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
def test_validate_invalid_A_G(A, G, channel_type):
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
def test_validate_valid_A_G(A, G, channel_type):
    """
    Test that the validate function does not raise an error for valid A and G values.
    """
    wrapper = PolarNR5GWrapper(A, G, channel_type)
    try:
        wrapper.validate()
    except ValueError:
        pytest.fail(f"validate() raised ValueError unexpectedly for channel type '{channel_type}'")

@pytest.mark.parametrize("A, G, channel_type, expected_params", [
    # Test cases for PUCCH
    (20, 100, 'PUCCH', {"A_min": 12, "A_max": 1706, "G_min": 31, "G_max": 8192, "pc_bits": 0, "pc_row_weight": 0}),
    (15, 200, 'PUCCH', {"A_min": 12, "A_max": 1706, "G_min": 18, "G_max": 8192, "pc_bits": 3, "pc_row_weight": 1}),
    (15, 20,  'PUCCH', {"A_min": 12, "A_max": 1706, "G_min": 18, "G_max": 8192, "pc_bits": 3, "pc_row_weight": 0}),
    (15, 300, 'PUCCH', {"A_min": 12, "A_max": 1706, "G_min": 18, "G_max": 8192, "pc_bits": 3, "pc_row_weight": 1}),
    # Test cases for PDCCH
    (10, 50, 'PDCCH', {"A_min": 1, "A_max": 140, "G_min": 25, "G_max": 8192, "pc_bits": 0, "pc_row_weight": 0}),
    # Test cases for PBCH
    (32, 864, 'PBCH', {"A_min": 32, "A_max": 32, "G_min": 864, "G_max": 864, "pc_bits": 0, "pc_row_weight": 0}),
])
def test_coding_parameters(A, G, channel_type, expected_params):
    """
    Test the coding parameters (A_min, A_max, G_min, G_max, pc_bits, pc_row_weight)
    for different channel types.
    """
    wrapper = PolarNR5GWrapper(A, G, channel_type)

    # Assertions for coding parameters
    assert wrapper.A_min == expected_params["A_min"], f"A_min mismatch for {channel_type}"
    assert wrapper.A_max == expected_params["A_max"], f"A_max mismatch for {channel_type}"
    assert wrapper.G_min == expected_params["G_min"], f"G_min mismatch for {channel_type}"
    assert wrapper.G_max == expected_params["G_max"], f"G_max mismatch for {channel_type}"
    assert wrapper.pc_bits == expected_params["pc_bits"], f"pc_bits mismatch for {channel_type}"
    assert wrapper.pc_row_weight == expected_params["pc_row_weight"], f"pc_row_weight mismatch for {channel_type}"



@pytest.mark.parametrize("A, G, channel_type, expected_n, expected_N", [
    # Test cases for PUCCH
    (100, 200, 'PUCCH', 8, 256),  # Example case where nmin = 5
    (1000, 2000, 'PUCCH', 10, 1024),  # Example case where nmax = 10
    # Test cases for PDCCH
    (50, 100, 'PDCCH', 7, 128),  # Example case for PDCCH
    (100, 500, 'PDCCH', 9, 512),  # Example case where nmax = 9
    # Test cases for PBCH
    (32, 864, 'PBCH', 9, 512),  # PBCH-specific case
])
def test_set_master_code_length_N(A, G, channel_type, expected_n, expected_N):
    """
    Test the _set_master_code_length_N method to ensure it calculates n and N correctly.
    """
    wrapper = PolarNR5GWrapper(A, G, channel_type)

    # Assertions for n and N
    assert wrapper.n == expected_n, f"n mismatch for {channel_type} with A={A}, G={G}: expected {expected_n}, got {wrapper.n}"
    assert wrapper.N == expected_N, f"N mismatch for {channel_type} with A={A}, G={G}: expected {expected_N}, got {wrapper.N}"


@pytest.mark.parametrize("G, A, channel_type, expected_N", [
    (24, 15, 'PUCCH', 32),
    (61, 45, 'PUCCH', 64),
    (76, 22, 'PUCCH', 128),
    (240, 117, 'PUCCH', 256),
    (325, 60, 'PUCCH', 512),
    (1009, 719, 'PUCCH', 1024),
    (43, 15, 'PDCCH', 64),
    (75, 8, 'PDCCH', 128),
    (228, 81, 'PDCCH', 256),
    (515, 66, 'PDCCH', 512),
    (864, 32, 'PBCH', 512),
])
def test_create_polar_encoder_matrix(G, A, channel_type, expected_N):
    """
    Test the create_polar_encoder_matrix method to ensure the generated NxN matrix
    matches the expected polar encoder matrix.
    """
    # Initialize the wrapper
    wrapper = PolarNR5GWrapper(A, G, channel_type)

    # Generate the polar encoder matrix using the wrapper
    generated_matrix = wrapper.matG_NxN

    # Calculate the expected polar encoder matrix
    base_matrix = np.array([[1, 0], [1, 1]], dtype=int)  # Base matrix for polar encoding
    log2_N = int(np.log2(expected_N))  # Calculate log2 of expected_N
    expected_matrix = base_matrix
    for _ in range(log2_N - 1):
        expected_matrix = np.kron(expected_matrix, base_matrix)  # Kronecker product
        # # Write the expected matrix to a file
        # with open(f"expected_matrix_{channel_type}_A{A}_G{G}.txt", "w") as file:
        #     for row in expected_matrix:
        #         file.write(" ".join(map(str, row)) + "\n")
    # Compare the generated matrix with the expected matrix
    assert np.array_equal(generated_matrix, expected_matrix), (
        f"Polar encoder matrix mismatch for A={A}, G={G}, Channel={channel_type}:\n"
        f"Expected:\n{expected_matrix}\nGenerated:\n{generated_matrix}"
    )


# @pytest.mark.parametrize("A, G, channel_type, expected_Qf1, expected_Qf2, expected_Qf3, expected_frozen_indices, expected_info_indices", [
@pytest.mark.parametrize("A, G, channel_type, expected_frozen_indices, expected_info_indices", [
    # Test cases for PUCCH
    (15, 24, 'PUCCH', 
    #  [24, 25, 26, 28, 27, 29, 30, 31], #Qf1
    #  [],  #Qf2
    #  [],  #Qf3
     [24, 25, 26, 27, 28, 29, 30, 31],  #frozen_indices
     [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), # info_indices [0, 1, 2] gone for pc_bits

    (45, 61, 'PUCCH', 
    #  [61, 62, 63], #Qf1
    #  [],  #Qf2
    #  [0, 1, 2, 4, 8],  #Qf3
     [0, 1, 2, 4, 8, 61, 62, 63],  #frozen_indices
     [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
      34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]),  # info_indices
    
    (22, 76, 'PUCCH', 
    #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 64, 65, 66, 67, 36, 37, 38, 39, 68, 69, 70, 71, 40, 41, 42, 43], #Qf1
    #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],  #Qf2
    #  [92, 105, 102, 90, 101, 89, 60, 86, 99, 58, 85, 78, 112, 57, 83, 54, 77, 53, 104, 75, 100, 88, 98, 84, 97, 56, 82, 76, 73, 81, 74, 96, 80, 72],  #Qf3
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 64, 65, 66, 67, 68, 69, 70, 71, 92, 105, 102, 90, 101, 89, 60, 86, 99, 58, 85, 78, 112, 57, 83, 54, 77, 53, 104, 75, 100, 88, 98, 84, 97, 56, 82, 76, 73, 81, 74, 96, 80, 72],  #frozen_indices
     [127, 126, 125, 123, 119, 111, 95, 124, 63, 122, 121, 118, 117, 110, 115, 109, 94, 107, 93, 103, 62, 120, 91, 61, 116, 87, 114, 59, 108, 79, 113, 55, 106])  # info_indices
])
# def test_set_subchannel_allocation(A, G, channel_type, expected_Qf1, expected_Qf2, expected_Qf3, expected_frozen_indices, expected_info_indices):
def test_set_subchannel_allocation(A, G, channel_type, expected_frozen_indices, expected_info_indices):
    """
    Test the _set_subchannel_allocation method to ensure it calculates Qf1, Qf2, Qf3,
    frozen_indices, and info_indices correctly.
    """
    # Initialize the wrapper
    wrapper = PolarNR5GWrapper(A, G, channel_type)

    # # Assertions for Qf1
    # assert sorted(wrapper.Qf1) == sorted(expected_Qf1), (
    #     f"Qf1 mismatch for {channel_type} with A={A}, G={G}: Expected: {sorted(expected_Qf1)} Generated: {sorted(wrapper.Qf1)}"
    # )

    # # Assertions for Qf2
    # assert sorted(wrapper.Qf2) == sorted(expected_Qf2), (
    #     f"Qf2 mismatch for {channel_type} with A={A}, G={G}: Expected: {expected_Qf2} Generated: {wrapper.Qf2}"
    # )

    # # Assertions for Qf3
    # assert sorted(wrapper.Qf3) == sorted(expected_Qf3), (
    #     f"Qf3 mismatch for {channel_type} with A={A}, G={G}: Expected: {expected_Qf3} Generated: {wrapper.Qf3}"
    # )

    # Assertions for frozen_indices
    assert sorted(wrapper.frozen_indices) == sorted(expected_frozen_indices), (
        f"frozen_indices mismatch for {channel_type} with A={A}, G={G}: Expected: {expected_frozen_indices} Generated: {wrapper.frozen_indices}"
    )

    # Assertions for info_indices
    assert sorted(wrapper.info_indices) == sorted(expected_info_indices), (
        f"info_indices mismatch for {channel_type} with A={A}, G={G}:\n"
        f"Expected: {expected_info_indices}\nGenerated: {wrapper.info_indices}"
    )

@pytest.mark.parametrize("G, A, channel_type, expected_N, expected_row_weights", [
    (24, 15, 'PUCCH', 32, [1,2,2,4,2,4,4,8,2,4,4,8,4,8,8,16,2,4,4,8,4,8,8,16,4,8,8,16,8,16,16,32]),
    (61, 45, 'PUCCH', 64, [1,2,2,4,2,4,4,8,2,4,4,8,4,8,8,16,2,4,4,8,4,8,8,16,4,8,8,16,8,16,16,32,2,4,4,8,4,8,8,16,4,8,8,16,8,16,16,32,4,8,8,16,8,16,16,32,8,16,16,32,16,32,32,64]),
])
def test_calculate_row_weights(A, G, channel_type, expected_N, expected_row_weights):
    """
    Test the _calculate_row_weights method to ensure it calculates row weights correctly.
    """
    # Initialize the wrapper
    wrapper = PolarNR5GWrapper(A, G, channel_type)
    mylist = range(expected_N)
    
    # Generate the row weights using the wrapper
    row_weights = wrapper._calculate_row_weights(mylist)

    # Convert generated row weights to Python integers
    row_weights = list(map(int, row_weights))

    # Compare the generated row weights with the expected row weights
    assert row_weights == expected_row_weights, (
        f"Row weights mismatch for A={A}, G={G}, Channel={channel_type}: Generated: {row_weights}"
    )



@pytest.mark.parametrize("A, G, channel_type, expected_parity_check_indices", [
    # Test cases for PUCCH
    (15, 24, 'PUCCH', [0, 1, 2]),  # Example parity check indices
    (45, 61, 'PUCCH', []),  # Example parity check indices
    (19,260, 'PUCCH',[242, 244, 248]), #N=256,RM=R,seg=0
    # Test cases for PDCCH
    (22, 76, 'PDCCH', []),  # Example parity check indices
    # Test cases for PBCH
    (32, 864, 'PBCH', []),  # Example parity check indices
])
def test_get_parity_check_indices(A, G, channel_type, expected_parity_check_indices):
    """
    Test the _get_parity_check_indices method to ensure it calculates parity check indices correctly.
    """
    # Initialize the wrapper
    wrapper = PolarNR5GWrapper(A, G, channel_type)

    # Get the parity check indices using the wrapper
    parity_check_indices = wrapper.pc_indices
    info_indices = wrapper.info_indices
    row_weights = wrapper.row_weights
    print("row_weights dumas: ", row_weights)
    # print("info_indices dumas: ", info_indices)
    min_row_weight_indices = wrapper.min_row_weight_indices
    print("seda: self.min_row_weight_indices: ", min_row_weight_indices)

    # Compare the generated parity check indices with the expected indices
    assert sorted(parity_check_indices) == sorted(expected_parity_check_indices), (
        f"Parity check indices mismatch for {channel_type} with A={A}, G={G}:\n"
        f"Expected: {expected_parity_check_indices}\nGenerated: {parity_check_indices}"
    )


@pytest.mark.parametrize("A, G, channel_type, expected_interleaver_indices", [
    # Test cases for PUCCH
    (15, 24, 'PUCCH', [0, 7, 13, 18, 22, 1, 8, 14, 19, 23, 2, 9, 15, 20, 3, 10, 16, 21, 4, 11, 17, 5, 12, 6]),  # Example interleaver indices
    (45, 61, 'PUCCH', [0, 11, 21, 30, 38, 45, 51, 56, 60, 1, 12, 22, 31, 39, 46, 52, 57, 2, 13, 23, 32, 40, 47, 53, 58, 3, 14, 24, 33, 41, 48, 54, 59, 4, 15, 25, 34, 42, 49, 55, 5, 16, 26, 35, 43, 50, 6, 17, 27, 36, 44, 7, 18, 28, 37, 8, 19, 29, 9, 20, 10]),  # Example interleaver indices
    # # Test cases for PDCCH
    # (22, 76, 'PDCCH', []),  # No channel interleaver for PDCCH
    # # Test cases for PBCH
    # (32, 864, 'PBCH', []),  # No channel interleaver for PBCH
])
def test_get_channel_interleaver_indices(A, G, channel_type, expected_interleaver_indices):
    """
    Test the _get_channel_interleaver_indices method to ensure it calculates
    channel interleaver indices correctly.
    """
    # Initialize the wrapper
    wrapper = PolarNR5GWrapper(A, G, channel_type)

    # Get the channel interleaver indices using the wrapper
    interleaver_indices = wrapper.channel_interleaver_indices

    # Compare the generated interleaver indices with the expected indices
    assert interleaver_indices == expected_interleaver_indices, (
        f"Channel interleaver indices mismatch for {channel_type} with A={A}, G={G}:\n"
        f"Expected: {expected_interleaver_indices}\nGenerated: {interleaver_indices}"
    )