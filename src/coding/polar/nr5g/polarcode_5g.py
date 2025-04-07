import math
from dataclasses import dataclass

@dataclass
class CRCSpec:
    name: str
    poly: int
    length: int

class PolarCodeChannelConfig:
    def __init__(self, A, G, channel_type):
        self.A = A                      # Number of information bits (no CRC)
        self.G = G                      # Rate-matched output length
        self.channel_type = channel_type

        # Always fixed
        self.nmin = 5
        self.Rmin = 1 / 8

        # Set channel-specific parameters
        self._set_channel_parameters()

        # Derived values
        self.K = self.A + self.crc.length
        self.n1 = self._compute_n1()
        self.n2 = self._compute_n2()
        self.n = max(self.nmin, min(self.n1, self.n2, self.nmax))
        self.N = 2 ** self.n
        self.R = self.K/self.E
        self.rm = self.decide_rate_matching_scheme()
        # print(self.summary())
        self.validate()

    def _set_channel_parameters(self):
        # Default values
        self.nmax = 10
        self.input_bits_interleaving = False
        self.channel_interleaver = False
        self.pc_bits = 0
        self.pc_row_weight = 0

        if self.channel_type in ('PUCCH', 'PUSCH'):
            self.A_min, self.A_max = 12, 1706
            if self.A >= 20:
                self.crc = CRCSpec("CRC11", 0x621, 11)
                self.G_min = 31
                if (1066 >= self.A >= 1013 and 1088 >= self.G >= 1036 ) or (1706 >= self.A >= 360 and 16385 >= self.G >= 1088):
                    self.G_max = 16384
                else:
                    self.G_max = 8192
            elif 12 <= self.A <= 19:
                self.crc = CRCSpec("CRC6", 0x21, 6)
                self.G_min = 18
                self.G_max = 8192
                self.pc_bits = 3
                self.pc_row_weight = 0 if (self.G - self.A <= 175) else 1
        elif self.channel_type == 'PDCCH':
            self.nmax = 9
            self.crc = CRCSpec("CRC24", 0xB2B117, 24)
            self.input_bits_interleaving = True
            self.A_min, self.A_max = 1, 140
            self.G_min, self.G_max = 25, 8192
        elif self.channel_type == 'PBCH':
            self.nmax = 9
            self.crc = CRCSpec("CRC24", 0xB2B117, 24)
            self.input_bits_interleaving = True
            self.A_min = self.A_max = 32
            self.G_min = self.G_max = 864
        else:
            raise ValueError(f"Unsupported channel type: {self.channel_type}")

    def decide_rate_matching_scheme(self):
        """
        Decide the rate matching scheme based on 5G NR rules.
        Returns: one of 'puncturing', 'shortening', or 'repetition'
        """
        if self.E <= self.N:
            if self.R <= 7/16:
                return 'puncturing'
            else:
                return 'shortening'
        else:
            return 'repetition'

    def _compute_n1(self):
        log2_G = math.ceil(math.log2(self.G))
        threshold1 = (9 / 8) * (2 ** (log2_G - 1))
        threshold2 = 9/16
        return log2_G - 1 if (self.G <= threshold1 and self.K/self.G < threshold2) else log2_G

    def _compute_n2(self):
        return math.ceil(math.log2(self.K / self.Rmin))

    def summary(self):
        return {
            "A": self.A,
            "G": self.G,
            "K": self.K,
            "channel_type": self.channel_type,
            "crc_name": self.crc.name,
            "crc_length": self.crc.length,
            "crc_poly": hex(self.crc.poly),
            "n1": self.n1,
            "n2": self.n2,
            "n": self.n,
            "N": self.N,
            "R": self.R,
            "nmin": self.nmin,
            "nmax": self.nmax,
            "input_bits_interleaving": self.input_bits_interleaving,
            "channel_interleaver": self.channel_interleaver,
            "pc_bits": self.pc_bits,
            "pc_row_weight": self.pc_row_weight,
            "G_range": (self.G_min, self.G_max),
            "A_range": (self.A_min, self.A_max),
            "rate_matching_scheme": (self.rm),
            "valid_config": self.validate()
        }

    def validate(self):
        if self.channel_type in ("PUCCH", "PUSCH"):
            if not (12 <= self.A <= 1706):
                raise ValueError(f"[UL] A={self.A} must be between 12 and 1706.")
            # if not (18 <= self.G <= 8192) and not self.segmentation:
            #     raise ValueError(f"[UL] G={self.G} must be between 18 and 8192.")
            # if self.segmentation and self.G > 16384:
            #     raise ValueError(f"[UL] G={self.G} exceeds 16384 with segmentation enabled.")

        elif self.channel_type == "PDCCH":
            if not (1 <= self.A <= 140):
                raise ValueError(f"[DL] A={self.A} must be between 1 and 140.")
            if self.A < 12:
                print(f"[Warning] A={self.A} is below 12. Zero-padding will be required later.")
            if not (25 <= self.G <= 8192):
                raise ValueError(f"[DL] G={self.G} must be between 25 and 8192.")

        elif self.channel_type == "PBCH":
            if self.A != 32 or self.G != 864:
                raise ValueError(f"[PBCH] A must be 32 and G must be 864 (got A={self.A}, G={self.G}).")

        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")

        return True
