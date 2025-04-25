import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from src.coding.crc.config.crc_config import CRCConfig
# from utils import determine_N  # assuming utils.py contains the logic
# from src.coding.polar.nr5g.polar_nr5g_wrapper import PolarNR5GWrapper

class constrainted_PolarNR5GWrapper:
    def __init__(self, A, G, channel_type):
        """
        Constrainted version of the original PolarNR5GWrapper class.
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

    def _set_segmentation_flag(self):
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
                self.crc = CRCConfig("CRC11", 11, 0, '5g')
                self.G_min = 31
                self.G_max = 16384 if self.segmentation else 8192
            elif 12 <= self.A <= 19:
                self.crc = CRCConfig("CRC6", 6, 0,'5g')
                self.G_min = 18
                self.G_max = 8192
                self.pc_bits = 3
                self.pc_row_weight = 0 if (self.G - self.A <= 175) else 1 #TODO: check this condition
        elif self.channel_type == 'PDCCH':
            self.crc = CRCConfig("CRC24", 24, 1,'5g')
            self.input_bits_interleaving = True
            self.A_min, self.A_max = 1, 140
            self.G_min, self.G_max = 25, 8192
        elif self.channel_type == 'PBCH':
            self.crc = CRCConfig("CRC24", 24, 1,'5g')
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

# Define a range of A and G values to sweep
A_values = list(range(12, 1706, 1)) 
G_values = list(range(32, 2048, 1)) 

# Build the N map
n_map = {}
for A in A_values:
    for G in G_values:
        if A >= G:
            continue  # invalid, can't have more info than total codeword
        # N = determine_N(A, G)
        config = constrainted_PolarNR5GWrapper(A, G, channel_type="PUCCH")
        n_map[(A, G)] = config.N

# Convert to DataFrame for heatmap
df = pd.DataFrame(index=A_values, columns=G_values)
for (A, G), N in n_map.items():
    df.at[A, G] = N


# Plot as heatmap
G_powers = [g for g in G_values if (g & (g - 1)) == 0]
A_powers = [a for a in A_values*2 if (a & (a - 1)) == 0]
plt.figure(figsize=(14, 8))
plt.imshow(df.astype(float), aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='N = 2^n')
plt.xticks(
    ticks=[G_values.index(g) for g in G_powers if g in G_values],
    labels=G_powers,
    rotation=90
)
plt.yticks(
    ticks=[A_values.index(a) for a in A_powers if a in A_values],
    labels=A_powers
)
plt.xlabel('G (Encoded Block Length)')
plt.ylabel('A (Information Block Length)')
plt.title('Map of (A, G) → N = 2^n')
plt.tight_layout()

# Save the plot to a file
plt.savefig("n_map_heatmap.png", dpi=300)
