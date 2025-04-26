import numpy as np
from src.tx.encoders.base_encoder import BaseEncoder
from src.coding.crc.crc_encoder import CRCEncoder  
from src.configs.config_polarcode import PolarCodeConfig

class PolarEncoder(BaseEncoder):
    """
    Concrete implementation of a polar encoder.
    """

    def __init__(self, config: PolarCodeConfig):
        super().__init__(A=config.len_k, G=config.len_n)
        self.crc_enable = config.crc.enable
        self.len_r = config.len_r
        self.len_logn = config.len_logn
        
        self.crc = CRCEncoder(config.crc) if config.crc.enable else None
        
        self.info_indices = config.info_indices
        self.info_bits_crc = np.zeros(config.len_k + self.len_r, dtype=np.uint8)
        # self.crc_indices = config.crc_indices
        self.vec_polar_non_info_indices = None
        self.matG_kxN = None
        self.matG_NxN = None
        self.matHt = None

        self.create_polar_matrices(int(self.len_logn))

    def _encode_np(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Main encoding logic.
        Appends CRC if enabled, then applies polar transform via matG_kxN.
        """
        if self.crc_enable:
            self.info_bits_crc = self.crc.encode_and_append(info_bits)
        else:
            self.info_bits_crc = info_bits

        return self.polar_encode(self.info_bits_crc)

    def polar_encode(self, uncoded_data: np.ndarray) -> np.ndarray:
        if self.matG_kxN is None:
            raise ValueError("The k-by-N generator matrix must be created first.")
        return (uncoded_data @ self.matG_kxN) % 2

    def create_polar_matrices(self, len_logn: int):
        """
        Creates generator matrices used for polar encoding.
        """
        matG_core = np.array([[1, 0], [1, 1]])
        matG = matG_core
        for _ in range(len_logn - 1):
            matG = np.kron(matG, matG_core)

        self.matG_NxN = matG
        self.matG_kxN = matG[sorted(self.info_indices)] # used for reduced encoder effort
        self.matG_Nxk = matG[:, sorted(self.info_indices)] # used for reduced decoder effort
        self.derive_parity_check_direct()

    def derive_parity_check_direct(self):
        """
        Derives the parity-check matrix H for diagnostics (optional).
        """
        N = self.matG_NxN.shape[1]
        all_indices = set(range(N))
        self.vec_polar_non_info_indices = list(all_indices - set(self.info_indices))
        self.matHt = self.matG_NxN[:, self.vec_polar_non_info_indices]

    def export_matrices(self):
        """
        Utility method to export internal matrices for debugging/visualization.
        """
        return {
            "matG_NxN": self.matG_NxN,
            "matG_kxN": self.matG_kxN,
            "matHt": self.matHt
        }
