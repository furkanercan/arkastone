import numpy as np
from src.tx.encoders.base_encoder import BaseEncoder
from src.coding.crc.config.crc_config import CRCConfig
from src.coding.crc.crc_encoder import CRCEncoder  # Example CRC encoder module

class PolarEncoder(BaseEncoder):
    """
    Concrete implementation of a polar encoder.
    """

    def __init__(self, code):
        super().__init__(A=code.len_k, G=code.len_n)
        self.en_crc = code.en_crc
        self.len_r = code.len_r
        
        self.crc_config = CRCConfig(
            name='crc',
            length=code.len_r,
            preload_val=0,  
            mode=code.crc_mode
        )
        self.crc = CRCEncoder(self.crc_config) if self.en_crc else None
        
        self.info_indices = code.info_indices
        self.crc_indices = code.crc_indices
        self.vec_polar_non_info_indices = None
        self.matG_kxN = None
        self.matG_NxN = None
        self.matHt = None

        self.create_polar_matrices(int(code.len_logn))

    def _encode_np(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Main encoding logic.
        Appends CRC if enabled, then applies polar transform via matG_kxN.
        """
        if self.en_crc:
            info_bits = self.crc.encode_and_append(info_bits)

        return self.polar_encode(info_bits)

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
        self.matG_kxN = matG[np.concatenate((self.info_indices, self.crc_indices))]
        self.derive_parity_check_direct()

    def derive_parity_check_direct(self):
        """
        Derives the parity-check matrix H for diagnostics (optional).
        """
        N = self.matG_NxN.shape[1]
        all_indices = set(range(N))
        self.vec_polar_non_info_indices = list(all_indices - set(self.info_indices) - set(self.crc_indices))
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
