from abc import ABC, abstractmethod
from typing import Union
from numpy.typing import NDArray
import numpy as np

class BaseEncoder(ABC):
    """
    Abstract base class for all encoder implementations.

    Responsibilities:
    - Defines a common interface for encoders via the encode() method.
    - Provides shared logic: input validation and type conversion.
    - Enforces that subclasses implement _encode_np(), which does the actual encoding logic.

    Input/output:
    - encode() takes a List[int] of length A (info bits)
    - encode() returns a List[int] of length G (coded bits)
    """

    def __init__(self, A: int, G: int, name: str = ""):
        """
        A: Number of input bits (info bits)
        G: Number of output bits (encoded bits)
        name: Optional name for display/debugging
        """
        if A <= 0 or G <= 0:
            raise ValueError("A and G must be positive integers.")
        if not isinstance(A, int) or not isinstance(G, int):
            raise TypeError("A and G must be integers.")

        self.A = A
        self.G = G
        self.name = name or self.__class__.__name__

    def encode(self, info_bits: Union[list[int], np.ndarray]) -> np.ndarray:
        """
        Main encoding entry point.
        Accepts list or np.ndarray of bits, validates, then calls subclass logic.
        Returns encoded bits as np.ndarray.
        """
        if isinstance(info_bits, list):
            info_bits = np.array(info_bits, dtype=np.uint8)
        elif not isinstance(info_bits, np.ndarray):
            raise TypeError(f"{self.name}: input must be list or np.ndarray.")

        self.check_input_length(info_bits)
        return self._encode_np(info_bits)

    @abstractmethod
    def _encode_np(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Subclasses must override this method to implement the actual encoding algorithm.
        It operates directly on NumPy arrays for performance.
        """
        pass

    def check_input_length(self, info_bits: Union[NDArray[np.int_], list]):
        """
        Helper method to validate input bit length matches expected A.
        """
        actual_len = len(info_bits)
        if actual_len != self.A:
            raise ValueError(f"{self.name}: Expected {self.A} bits, got {actual_len}.")

    def reset(self):
        """
        Optional: Reset internal state if the encoder is stateful.
        No-op for stateless encoders.
        """
        pass

    def __repr__(self):
        return f"{self.name}(A={self.A}, G={self.G})"
