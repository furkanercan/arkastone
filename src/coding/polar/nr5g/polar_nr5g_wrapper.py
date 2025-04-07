import math
from dataclasses import dataclass
from src.coding.polar.nr5g.polar_nr5g_encoder_chains import pucch_encoder, pusch_encoder, pdcch_encoder, pbch_encoder
from src.coding.polar.nr5g.polar_nr5g_decoder_chains import pucch_decoder, pusch_decoder, pdcch_decoder, pbch_decoder


@dataclass
class CRCSpec:
    name: str
    poly: int
    length: int


class PolarNR5GWrapper:
    def __init__(self, A, G, channel_type):
        """
        Initializes the PolarNR5GWrapper class with the given parameters and sets up
        the necessary coding and segmentation configurations.
        Args:
            A (int): Number of information bits (excluding CRC).
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
        self.G = G                      # Rate-matched output length after concatenation
        self.E = G                      # Rate-matched output length before concatenation
        self.channel_type = channel_type

        self._set_segmentation_flag()
        self._set_coding_parameters()

        if(self.segmentation):
            self.E = self.G // 2 # Set E to half of G for segmentation
            self.Abar = self.A // 2 # Set Abar to half of original A for segmentation
            self.K = self.Abar + self.crc.length
        else:
            self.K = self.A + self.crc.length
        
        self._set_master_code_length_N()
        self.R = self.K/self.E
        self._set_rate_matching_scheme()

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
        guidance for this implementation. However, further research is required to ensure 
        proper assignment and validation of the segmentation logic.
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
                self.crc = CRCSpec("CRC11", 0x621, 11)
                self.G_min = 31
                self.G_max = 16384 if self.segmentation else 8192
            elif 12 <= self.A <= 19:
                self.crc = CRCSpec("CRC6", 0x21, 6)
                self.G_min = 18
                self.G_max = 8192
                self.pc_bits = 3
                self.pc_row_weight = 0 if (self.G - self.A <= 175) else 1
        elif self.channel_type == 'PDCCH':
            self.crc = CRCSpec("CRC24", 0xB2B117, 24)
            self.input_bits_interleaving = True
            self.A_min, self.A_max = 1, 140
            self.G_min, self.G_max = 25, 8192
        elif self.channel_type == 'PBCH':
            self.crc = CRCSpec("CRC24", 0xB2B117, 24)
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

    def encode(self, input_bits):
        """
        Encodes the input bits using the appropriate encoder chain based on channel type.
        Args:
            input_bits (list[int]): List of input bits to encode.
        Returns:
            list[int]: Encoded bits.
        """
        if self.channel_type == 'PUCCH':
            return pucch_encoder(input_bits, self)
        elif self.channel_type == 'PUSCH':
            return pusch_encoder(input_bits, self)
        elif self.channel_type == 'PDCCH':
            return pdcch_encoder(input_bits, self)
        elif self.channel_type == 'PBCH':
            return pbch_encoder(input_bits, self)
        else:
            raise ValueError(f"Unsupported channel type: {self.channel_type}")

    def decode(self, received_bits):
        """
        Decodes the received bits using the appropriate decoder chain based on channel type.
        Args:
            received_bits (list[int]): List of received bits to decode.
        Returns:
            list[int]: Decoded bits.
        """
        if self.channel_type == 'PUCCH':
            return pucch_decoder(received_bits, self)
        elif self.channel_type == 'PUSCH':
            return pusch_decoder(received_bits, self)
        elif self.channel_type == 'PDCCH':
            return pdcch_decoder(received_bits, self)
        elif self.channel_type == 'PBCH':
            return pbch_decoder(received_bits, self)
        else:
            raise ValueError(f"Unsupported channel type: {self.channel_type}")