import math
import numpy as np
from dataclasses import dataclass
# from src.tx.nr5g.polar.polar_nr5g_encoder_chains import pucch_encoder, pusch_encoder, pdcch_encoder, pbch_encoder
# from src.tx.nr5g.polar.polar_nr5g_decoder_chains import pucch_decoder, pusch_decoder, pdcch_decoder, pbch_decoder
from src.tx.nr5g.polar.components.subblock_interleaver import subblock_interleaver
from src.configs.config_crc import CRCConfig
from src.tx.nr5g.polar.config.pucch_config import PUCCHConfig


class PolarNR5GWrapper:
    def __init__(self, A, G, channel_type):
        """
        Initializes the PolarNR5GWrapper class with the given parameters and sets up
        the necessary coding and segmentation configurations.
        Args:
            A (int): Number of informatio_compute_n1(self):

_compute_n2(self):

_set_master_code_length_N(self):n bits (excluding CRC).
            G (int): Rate-matched output length after concatenation.
            channel_type (str): Type of the communication channel.
        Attributes:
            A (int): Number of information bits (excluding CRC).
            G (int): Rate-matched output length after concatenation.
            E (int): Rate-matched output length before concatenation.
            channel_type (str): Type of the communication channel.
            segmentation (bool): Flag indicating whether segmentation is enabled.
            Abar (int): Half of the original A value, used when segmentation is enabled.
            K (int): Total number of bits after adding CRC.
            N (int): Master code length, derived based on coding parameters.
            R (float): Code rate, calculated as K/E.
        """
        self.A = A               # Number of information bits (no CRC)
        self.G = G               # Rate-matched output length after concatenation
        self.E = G               # Rate-matched output length before concatenation
        self.channel_type = channel_type

        self._set_segmentation_flag()
        self._set_coding_parameters()
        self.validate()          # Validate A and G against the specified limits

        if(self.segmentation):
            self.E = self.G // 2 # Set E to half of G for segmentation
            self.Abar = self.A // 2 # Set Abar to half of original A for segmentation
            self.K = self.Abar + self.crc.length
        else:
            self.K = self.A + self.crc.length
        
        self._set_master_code_length_N()
        self.logN = int(math.log2(self.N))
        self._get_reliability_indices()
        vec_range_N = list(range(self.N)) # temp [0, 1, 2, ..., N-1] for interleaving
        self.interleaved_indices = subblock_interleaver(vec_range_N, self.N)
        self.R = self.K/self.E
        self._set_rate_matching_scheme()
        self._set_subchannel_allocation() #get frozen and (initial) info indices, i.e. info+crc+pc indices
        self._create_polar_encoder_matrix() # create the NxN polar matrix
        self._get_parity_check_indices() # get parity check indices and fine-tuned info indices for info and CRC bits
        self._create_polar_encoder_matrix_optimized() # create the kxN polar matrix
        self._get_channel_interleaver_indices()
        self.info_indices = sorted(self.info_indices)
        self.pc_indices = sorted(self.pc_indices)
        self.frozen_indices = sorted(self.frozen_indices)

    def _set_segmentation_flag(self):
        """
        Sets the segmentation flag based on the channel type and specific conditions.

        This method determines whether segmentation is required for the given channel type 
        ('PUCCH' or 'PUSCH') and parameters `A` and `G`. The segmentation flag is set to 
        `True` if the conditions specified in the method are met; otherwise, it is set to `False`.

        References:
        - Valerio's paper: "Design of Polar Codes for 5G New Radio" (IEEE)
        - 3GPP TS 38.212: "Multiplexing and channel coding" (3GPP)
        - Egilmez paper: "The Development, Operation and Performance of the 5G Polar Codes" (IEEE)

        Note:
        We take the Egilmez paper as the primary reference since it provides more specific 
        guidance for this implementation. However, Valerio's formulation is also valid.
        """
        if self.channel_type in ('PUCCH', 'PUSCH'):
            if (1066 >= self.A >= 1013 and 1088  >= self.G >= 1036) or \
               (1706 >= self.A >= 360  and 16385 >= self.G >= 1088):
                self.segmentation = True
            else:
                self.segmentation = False
        else:
            self.segmentation = False

    def _set_coding_parameters(self):
        # Default values
        self.input_bits_interleaving = False
        self.channel_interleaver = False
        self.pc_bits = 0
        self.pc_row_weight = 0

        if self.channel_type in ('PUCCH', 'PUSCH'):
            self.A_min, self.A_max = 12, 1706
            if self.A >= 20:
                self.crc = CRCConfig(1, "CRC11", 11, 0, '5g')
                self.G_min = 31
                self.G_max = 16384 if self.segmentation else 8192
            elif 12 <= self.A <= 19:
                self.crc = CRCConfig(1, "CRC6", 6, 0,'5g')
                self.G_min = 18
                self.G_max = 8192
                self.pc_bits = 3
                self.pc_row_weight = 0 if (self.G - self.A <= 175) else 1 #TODO: check this condition
        elif self.channel_type == 'PDCCH':
            self.crc = CRCConfig(1, "CRC24", 24, 1,'5g')
            self.input_bits_interleaving = True
            self.A_min, self.A_max = 1, 140
            self.G_min, self.G_max = 25, 8192
        elif self.channel_type == 'PBCH':
            self.crc = CRCConfig(1, "CRC24", 24, 1,'5g')
            self.input_bits_interleaving = True
            self.A_min = self.A_max = 32
            self.G_min = self.G_max = 864
        else:
            raise ValueError(f"Unsupported channel type: {self.channel_type}")

    def _compute_n1(self):
        log2_E = math.ceil(math.log2(self.E))
        condition1 = (9 / 8) * (2 ** (log2_E - 1))
        condition2 = 9/16
        return log2_E - 1 if (self.E <= condition1 and self.K/self.E < condition2) else log2_E

    def _compute_n2(self):
        Rmin = 1/8
        return math.ceil(math.log2(self.K / Rmin))
    
    def _set_master_code_length_N(self):
        nmin = 5
        nmax = 9 if self.channel_type in ('PDCCH', 'PBCH') else 10
        n1 = self._compute_n1()
        n2 = self._compute_n2()
        self.n = max(nmin, min(n1, n2, nmax))
        self.N = 2 ** self.n
    
    def _get_reliability_indices(self):
        """
        Returns the reliability indices based on the master reliability index.
        If N is 1024, the full master reliability index is returned.
        Otherwise, a subset where values >= N are redacted is returned.
        """
        master_reliability_index = [
            1023, 1022, 1021, 1019, 1015, 1007, 1020, 991, 1018, 1017, 1014, 1006, 895, 1013, 1011, 959, 1005, 990, 1003,
            989, 767, 1016, 999, 1012, 987, 958, 983, 957, 1010, 1004, 955, 1009, 894, 975, 893, 1002, 951, 1001, 988,
            511, 766, 998, 891, 943, 986, 997, 985, 887, 956, 765, 995, 927, 982, 981, 879, 954, 974, 763, 953, 979, 510,
            1008, 759, 863, 950, 892, 1000, 973, 949, 509, 890, 971, 996, 942, 751, 984, 889, 507, 947, 831, 886, 967,
            941, 764, 926, 980, 994, 939, 885, 993, 735, 878, 925, 503, 762, 883, 978, 935, 703, 495, 952, 877, 761, 972,
            923, 977, 948, 758, 862, 875, 919, 970, 757, 861, 508, 969, 750, 946, 479, 888, 639, 871, 911, 830, 940, 859,
            755, 966, 945, 749, 506, 884, 938, 965, 829, 734, 924, 855, 505, 747, 963, 937, 882, 934, 827, 733, 447, 992,
            847, 876, 501, 921, 702, 494, 881, 760, 743, 933, 502, 918, 874, 922, 823, 731, 499, 860, 756, 931, 701, 873,
            493, 727, 917, 870, 976, 815, 910, 383, 968, 478, 858, 754, 699, 491, 869, 944, 748, 638, 915, 477, 719, 909,
            964, 255, 799, 504, 857, 854, 753, 828, 746, 695, 487, 907, 637, 867, 853, 475, 936, 962, 446, 732, 826, 745,
            846, 500, 825, 903, 687, 932, 635, 471, 445, 742, 880, 498, 730, 851, 822, 382, 920, 845, 741, 443, 700, 729,
            631, 492, 872, 961, 726, 821, 930, 497, 381, 843, 463, 916, 739, 671, 623, 490, 929, 439, 814, 819, 868, 752,
            914, 698, 725, 839, 856, 476, 813, 718, 908, 486, 723, 866, 489, 607, 431, 697, 379, 811, 798, 913, 575, 717,
            254, 694, 636, 474, 807, 715, 906, 797, 693, 865, 960, 852, 744, 634, 473, 795, 905, 485, 415, 483, 470, 444,
            375, 850, 740, 686, 902, 824, 691, 253, 711, 633, 844, 685, 630, 901, 367, 791, 928, 728, 820, 849, 783, 670,
            899, 738, 842, 683, 247, 469, 441, 442, 462, 251, 737, 438, 467, 351, 629, 841, 724, 679, 669, 496, 461, 818,
            380, 437, 627, 622, 459, 378, 239, 488, 667, 838, 430, 484, 812, 621, 319, 817, 435, 377, 696, 722, 912, 606,
            810, 864, 716, 837, 721, 714, 809, 796, 455, 472, 619, 835, 692, 663, 223, 414, 904, 427, 806, 482, 632, 713,
            690, 848, 605, 373, 252, 794, 429, 710, 684, 615, 805, 900, 655, 468, 366, 603, 413, 574, 481, 371, 250, 793,
            466, 423, 374, 689, 628, 440, 365, 709, 789, 803, 411, 573, 682, 249, 460, 790, 668, 599, 350, 707, 246, 681,
            465, 571, 626, 436, 407, 782, 191, 127, 363, 620, 666, 458, 245, 349, 677, 434, 678, 591, 787, 399, 457, 359,
            238, 625, 840, 567, 736, 665, 428, 376, 781, 898, 618, 675, 318, 454, 662, 243, 897, 347, 836, 816, 720, 433,
            604, 617, 779, 808, 661, 834, 712, 804, 833, 559, 237, 453, 426, 222, 317, 775, 372, 343, 412, 235, 543, 614,
            451, 425, 422, 613, 370, 221, 315, 480, 335, 659, 654, 364, 190, 369, 248, 653, 688, 231, 410, 602, 611, 802,
            792, 421, 651, 601, 598, 708, 311, 219, 572, 597, 788, 570, 409, 590, 362, 801, 680, 464, 406, 419, 348, 647,
            786, 215, 589, 706, 361, 676, 566, 189, 595, 244, 569, 303, 405, 358, 456, 346, 398, 565, 242, 126, 705, 780,
            587, 624, 664, 236, 187, 357, 432, 785, 558, 674, 207, 403, 397, 452, 345, 563, 778, 241, 316, 342, 616, 660,
            557, 125, 234, 183, 287, 355, 583, 673, 395, 424, 314, 220, 777, 341, 612, 658, 123, 175, 774, 555, 233, 334,
            542, 450, 313, 391, 230, 652, 368, 218, 339, 600, 119, 333, 657, 610, 773, 541, 310, 420, 159, 229, 650, 551,
            596, 609, 408, 217, 449, 188, 309, 214, 331, 111, 539, 360, 771, 649, 302, 418, 594, 896, 227, 404, 646, 186,
            588, 832, 568, 213, 417, 301, 307, 356, 402, 800, 564, 327, 95, 206, 240, 535, 593, 645, 586, 344, 396, 185,
            401, 211, 354, 299, 585, 286, 562, 643, 182, 205, 124, 232, 285, 295, 181, 556, 582, 527, 394, 340, 63, 203,
            561, 353, 448, 122, 283, 393, 581, 554, 174, 390, 704, 312, 338, 228, 179, 784, 199, 553, 121, 173, 389, 540,
            579, 332, 118, 672, 550, 337, 158, 279, 271, 416, 216, 308, 387, 538, 549, 226, 330, 776, 171, 212, 117, 110,
            329, 656, 157, 772, 306, 326, 225, 167, 115, 537, 534, 184, 109, 300, 547, 305, 210, 155, 533, 325, 352, 608,
            400, 298, 204, 94, 648, 284, 209, 151, 180, 107, 770, 297, 392, 323, 592, 202, 644, 93, 294, 178, 103, 143,
            282, 62, 336, 201, 120, 172, 198, 769, 584, 91, 388, 293, 177, 526, 278, 281, 642, 525, 531, 61, 170, 116,
            197, 87, 156, 277, 114, 560, 169, 59, 291, 580, 275, 523, 641, 270, 195, 552, 519, 166, 224, 578, 108, 269,
            79, 154, 113, 548, 577, 536, 328, 55, 106, 165, 153, 150, 386, 208, 324, 546, 385, 267, 47, 92, 163, 296, 304,
            105, 102, 149, 263, 532, 322, 292, 545, 90, 200, 31, 321, 530, 142, 176, 147, 101, 141, 196, 524, 529, 290,
            89, 280, 60, 86, 99, 139, 168, 58, 522, 276, 85, 194, 289, 78, 135, 112, 521, 57, 83, 54, 518, 274, 268, 768,
            164, 77, 152, 193, 53, 162, 104, 517, 273, 266, 75, 46, 148, 51, 640, 100, 45, 576, 161, 265, 262, 71, 146,
            30, 140, 88, 515, 98, 43, 29, 261, 145, 138, 84, 259, 39, 97, 27, 56, 82, 137, 76, 384, 134, 23, 52, 133, 320,
            15, 73, 50, 81, 131, 44, 70, 544, 192, 528, 288, 520, 160, 272, 74, 49, 516, 42, 69, 28, 144, 41, 67, 96, 514,
            38, 264, 260, 136, 22, 25, 37, 80, 513, 26, 258, 35, 132, 21, 257, 72, 14, 48, 13, 19, 130, 68, 40, 11, 512,
            66, 129, 7, 36, 24, 34, 256, 20, 65, 33, 12, 128, 18, 10, 17, 6, 9, 64, 5, 3, 32, 16, 8, 4, 2, 1, 0
        ]
        if self.N == 1024:
            self.reliability_indices = master_reliability_index
        else:
            self.reliability_indices = [x for x in master_reliability_index if x < self.N]
        
        # print(self.reliability_indices)


    def _set_rate_matching_scheme(self):
        """
        Decide the rate matching scheme based on 5G NR rules.
        Returns: one of 'puncturing', 'shortening', or 'repetition'
        """
        if self.E <= self.N:
            if self.R <= 7/16:
                self.rm = 'puncturing'
            else:
                self.rm = 'shortening'
        else:
            self.rm = 'repetition'

    def _set_subchannel_allocation(self):
        """
        Calculate the set of frozen and non-frozen indices for polar coding based on the given parameters.
        """

        # Initialize frozen index lists
        Qf1 = []
        Qf2 = []
        Qf3 = []

        # Calculate Qf1 based on the rate matching scheme
        if self.rm == 'puncturing':
            Qf1 = self.interleaved_indices[:self.N - self.E]  # First N-E indices
        elif self.rm == 'shortening':
            Qf1 = self.interleaved_indices[-(self.N - self.E):]  # Last N-E indices
        elif self.rm == 'repetition':
            Qf1 = []  # Empty list for repetition

        # Print Qf1 for debugging
        # print(f"Qf1 (based on rate matching scheme): {Qf1}")

        # Calculate Qf2 as [0, ..., T] if the scheme is 'puncturing'
        if self.rm == 'puncturing':
            # Calculate T based on the given formula
            if self.E >= (3 / 4) * self.N:
                T = math.ceil((3 / 4) * self.N - self.E / 2) - 1
            else:
                T = math.ceil((9 / 16) * self.N - self.E / 4) - 1
            Qf2 = list(range(T + 1))  # [0, ..., T]
        else:
            Qf2 = []  # Empty list for others

        # Print Qf2 for debugging
        # print(f"Qf2 (calculated based on T): {Qf2}")

        # Qf3 is derived from the remaining indices after excluding Qf1 and Qf2
        remaining_indices = [x for x in self.reliability_indices if x not in Qf1 and x not in Qf2]
        # print(f"Remaining Indices (after excluding Qf1 and Qf2): {remaining_indices}")
        Qf3 = remaining_indices[(self.K + self.pc_bits):]  # Exclude the first K + n_pc indices

        # Print Qf3 for debugging
        # print(f"Qf3 (remaining indices after excluding Qf1 and Qf2): {Qf3}")

        # Combine Qf1, Qf2, and Qf3 into frozen_indices
        self.frozen_indices = list(set(Qf1 + Qf2 + Qf3))
        # print(f"Frozen Indices (Qf1 + Qf2 + Qf3): {self.frozen_indices}")
        
        # Calculate info_indices as the complement of frozen_indices with respect to self.reliability_indices
        # Important note: current info_indices include parity check indices as well.
        # Later in _get_parity_check_indices() we will remove them from info_indices.
        self.info_indices = [x for x in self.reliability_indices if x not in self.frozen_indices]
        # print(f"Information Indices (complement of frozen_indices): {self.info_indices}")


    def _create_polar_encoder_matrix(self):
        """
        Creates the polar matrices:
            - matG_kxN: The generator matrix in k-by-N form.
            - matG_NxN: The generator matrix in N-by-N form.
            - matHt   : The transposed parity-check matrix in N-by-(N-k) form. (TODO if needed)

        Raises:
            TypeError: If self.logN is not a positive integer.
        """
        if not isinstance(self.logN, int) or self.logN <= 0:
            raise TypeError("self.logN must be a positive integer")
        matG_core = np.array([[1, 0], [1, 1]])
        matG = matG_core  # Core matrix as the initial value
        for _ in range(self.logN-1):
            matG = np.kron(matG, matG_core)

        self.matG_NxN = matG                # Full NxN G matrix

    def _create_polar_encoder_matrix_optimized(self):
        """
        Creates the k-by-N polar matrix for efficient computing.
        The matrix is pruned to include only the information, crc and parity check indices.
        IMPORTANT: If this matrix is to be used for encoding, make sure that the pc bits are 
        placed correctly amongst information indices based on their reliability indices.
        """
        self.matG_kxN = self.matG_NxN[list(sorted(self.info_indices + self.pc_indices))] # Pruned G matrix (kxN)
        
    def _get_parity_check_indices(self):
        """
        
        """
        self.row_weights = []
        self.min_row_weight_indices = []
        if(self.pc_bits == 0):
            self.pc_indices = []
        else: # pc_bits == 3
            if(self.pc_row_weight == 1):
                # print(f"Info indices: {self.info_indices}")
                # Assign the two parity check bits to the next two lowest reliability indices
                self.pc_indices = self.info_indices[-2:]
                # print(f"pc_indices first 2: {self.pc_indices}")
                info_indices_search_space = self.info_indices[:-3]
                # print(f"Info indices search space: {info_indices_search_space}")
                self.row_weights = self._calculate_row_weights(info_indices_search_space) # Calculate the row weights of the generator matrix
                # print(f"Row weights: {self.row_weights}")
                # Find the list of indices of the min row weights
                min_weight = min(self.row_weights)
                # print(f"Min row weight: {min_weight}")
                min_weight_indices = [i for i, weight in enumerate(self.row_weights) if weight == min_weight]
                # print(f"Min row weight indices: {min_weight_indices}")
                self.min_row_weight_indices = [info_indices_search_space[i] for i in min_weight_indices]
                # print(f"Min row weight indices (reliability): {self.min_row_weight_indices}")
                self.min_row_weight_indices = [self.min_row_weight_indices[-1]]  # Keep only the last item

                # Among them, find the one with the min reliability 
                best_index = self.min_row_weight_indices[0]

                # Append last parity check bit to that best index
                self.pc_indices.append(best_index)

            else:
                # Assign all three parity check bits to the 3 lowest reliability indices
                self.pc_indices = self.info_indices[-3:]
        
        # Remove the parity check indices from the info indices
        self.info_indices = [x for x in self.info_indices if x not in self.pc_indices]
                
    def _calculate_row_weights(self, info_indices_search_space):
        """
        Calculates the row weights of the generator matrix.
        Returns:
            list[int]: List of row weights for each row in the generator matrix.
        """
        # Calculate the row weights based on the generator matrix
        row_weights = np.sum(self.matG_NxN, axis=1)
        # print("Row Weights:")
        # print(row_weights)
        return [row_weights[i] for i in info_indices_search_space]

    def _get_channel_interleaver_indices(self):
        """
        Determines the channel interleaver indices based on the value of E.

        The method calculates the minimum triangular number T such that the sum of the first T natural numbers
        (T * (T + 1) / 2) is greater than or equal to E. It then constructs a T x T matrix, filling it row-by-row
        with indices from 0 to E-1. The matrix is read column-wise to generate the interleaver indices.
        """
        self.T = 0
        while self.E > self.T * (self.T + 1) // 2:
            self.T += 1
        
        # Create a self.T by self.T matrix and fill it row-by-row with indices starting from 0
        self.matrix = np.full((self.T, self.T), -1)  # Initialize with -1 to handle missing indices
        index = 0
        for i in range(self.T):
            for j in range(self.T-i):  # Stop one index earlier for each row
                if index >= self.E:  # Stop entirely when index reaches E
                    break
                self.matrix[i, j] = index
                index += 1
            if index >= self.E:  # Break the outer loop as well
                break
        # print("Channel interleaver matrix:")
        # print(self.matrix.tolist())
        # Read the indices column-wise
        self.channel_interleaver_indices = [x for x in self.matrix.T.flatten() if x != -1]

    def validate(self):
        """
        Validates the A and G values against the specified limits (A_min/A_max, G_min/G_max).
        Raises:
            ValueError: If A or G is out of the valid range.
        """
        if not (self.A_min <= self.A <= self.A_max):
            raise ValueError(
                f"Channel type '{self.channel_type}': A ({self.A}) is out of range: [{self.A_min}, {self.A_max}]"
            )
        if not (self.G_min <= self.G <= self.G_max):
            raise ValueError(
                f"Channel type '{self.channel_type}': G ({self.G}) is out of range: [{self.G_min}, {self.G_max}]"
            )
        if not (self.A < self.G): #TODO: add by how much A should be smaller than G later.
            raise ValueError(
                f"Channel type '{self.channel_type}': A ({self.A}) must be smaller than G ({self.G})."
            )

    # def encode(self, input_bits):
    #     """
    #     Encodes the input bits using the appropriate encoder chain based on channel type.
    #     Args:
    #         input_bits (list[int]): List of input bits to encode.
    #     Returns:
    #         list[int]: Encoded bits.
    #     """
    #     if self.channel_type == 'PUCCH':
    #         return pucch_encoder(input_bits, self)
    #     elif self.channel_type == 'PUSCH':
    #         return pusch_encoder(input_bits, self)
    #     elif self.channel_type == 'PDCCH':
    #         return pdcch_encoder(input_bits, self)
    #     elif self.channel_type == 'PBCH':
    #         return pbch_encoder(input_bits, self)
    #     else:
    #         raise ValueError(f"Unsupported channel type: {self.channel_type}")

    # def decode(self, received_bits):
    #     """
    #     Decodes the received bits using the appropriate decoder chain based on channel type.
    #     Args:
    #         received_bits (list[int]): List of received bits to decode.
    #     Returns:
    #         list[int]: Decoded bits.
    #     """
    #     if self.channel_type == 'PUCCH':
    #         return pucch_decoder(received_bits, self)
    #     elif self.channel_type == 'PUSCH':
    #         return pusch_decoder(received_bits, self)
    #     elif self.channel_type == 'PDCCH':
    #         return pdcch_decoder(received_bits, self)
    #     elif self.channel_type == 'PBCH':
    #         return pbch_decoder(received_bits, self)
    #     else:
    #         raise ValueError(f"Unsupported channel type: {self.channel_type}")

    def summary(self):
        """
        Returns a summary of the key parameters of the PolarNR5GWrapper instance.
        Returns:
            dict: A dictionary containing the key parameters and their values.
        """
        return {
            "A": self.A,
            "G": self.G,
            "E": self.E,
            "channel_type": self.channel_type,
            "segmentation": self.segmentation,
            "Abar": getattr(self, "Abar", None),  # Only present if segmentation is enabled
            "K": self.K,
            "N": self.N,
            "R": self.R,
            "rate_matching_scheme": self.rm,
            "crc": {
                "name": self.crc.name,
                "poly": hex(self.crc.poly),
                "length": self.crc.length,
            },
        }
    
    def _generate_pucch_encoder_config(self) -> PUCCHConfig:
        return PUCCHConfig(
            A=self.A,
            K=self.K,
            N=self.N,
            E=self.E,
            G=self.G,
            # Abar = self.Abar,
            rm=self.rm,
            seg = self.segmentation,
            crc_config=self.crc,
            pc_indices = self.pc_indices,
            info_indices= self.info_indices,
            frozen_indices= self.frozen_indices,
            channel_interleaved_indices = self.channel_interleaver_indices,
            Gmat_kxN = self.matG_kxN,
            # ...
        )