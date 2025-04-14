import numpy as np
import math


class Modulator:
    def __init__(self, config):
        self.scheme = config["type"].lower()
        self.validate_scheme()

        if self.scheme == "bpsk":
            self.normalization_factor = 1
            self.num_constellations = 2
        elif self.scheme == "qpsk":
            self.normalization_factor = 1/np.sqrt(2)
            self.num_constellations = 4
        elif self.scheme == "16qam":
            self.normalization_factor = 1/np.sqrt(10)
            self.num_constellations = 16
        elif self.scheme == "64qam":
            self.normalization_factor = 1/np.sqrt(42)
            self.num_constellations = 64
        elif self.scheme == "256qam":
            self.normalization_factor = 1/np.sqrt(85)
            self.num_constellations = 256
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.scheme}")

        self.log_num_constellations = int(np.log2(self.num_constellations))

    def validate_scheme(self):
        valid_schemes = ["bpsk", "qpsk", "16qam", "64qam", "256qam"]
        if self.scheme not in valid_schemes:
            raise ValueError(f"Unsupported modulation scheme: {self.scheme}. Supported schemes: {valid_schemes}")

    def modulate(self, modulated_data, bool_data):
        """
        Modulate the given data based on the selected scheme.
        Args:
            data (np.ndarray): Input bits to modulate (1D array of 0s and 1s).

        Returns:
            np.ndarray: Modulated symbols.
        """
        
        if self.scheme == "bpsk":
            self.mod_bpsk(modulated_data, bool_data)
        elif self.scheme == "qpsk":
            self.mod_qpsk(modulated_data, bool_data)
        elif self.scheme == "16qam":
            self.mod_mqam(modulated_data, bool_data, 16)
        elif self.scheme == "64qam":
            self.mod_mqam(modulated_data, bool_data, 64)
        elif self.scheme == "256qam":
            self.mod_mqam(modulated_data, bool_data, 256)
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.scheme}")

    def mod_bpsk(self, modulated_data, bool_data):
        """
        Perform BPSK modulation.

        Args:
            bool_data (np.ndarray): Array of binary input bits (0s and 1s).

        Returns:
            np.ndarray: Array of modulated values (-1 and +1).
        
        Raises:
            ValueError: If bool_data contains values other than 0 or 1.
        """
        if not np.all(np.logical_or(bool_data == 0, bool_data == 1)):
            raise ValueError("input data to modulator must be Boolean.")
        
        bool_data = np.asarray(bool_data) # Ensure data type
        
        modulated_data[:] = 1 - 2 * bool_data #Special case: normalization factor is 1.

    def mod_qpsk(self, modulated_data, bool_data):
        """QPSK modulation: Map bits to quadrature phase"""
        
        if len(bool_data) % 2 != 0:
            raise ValueError("Number of input bits must be even for QPSK.") #Solve this later.
        
        bool_data = bool_data.reshape(-1, 2)  # Reshape to pairs of bits
        mapping = {0:  1, 
                   1: -1}
        i = np.array([mapping[bit] for bit in bool_data[:, 0]])  # In-phase
        q = np.array([mapping[bit] for bit in bool_data[:, 1]])  # Quadrature
        modulated_data[:] =  (i + 1j * q)*self.normalization_factor


    def mod_mqam(self, modulated_data, bool_data, m):
        """Generalized M-QAM modulation"""
        
        # int_list = [int(x) for x in bool_data]
        # print(int_list)  # Output: [1, 0, 1, 1, 0]
        
        ## CHECK these during init, not here.
        if np.log2(m) % 1 != 0:
            raise ValueError("Modulation order M must be a power of 2.") #Solve this later.

        num_bits_per_symbol = int(np.log2(m))
        if len(bool_data) % num_bits_per_symbol != 0:
            raise ValueError(f"Number of input bits must be divisible by {num_bits_per_symbol} for {m}-QAM.")

        # levels = np.arange(np.sqrt(m) - 1, -np.sqrt(m), step=-2) ##LUT
        levels = np.array([-3, -1, 3, 1], dtype=int)
        # print(levels)
        
        bool_data = bool_data.reshape(-1, num_bits_per_symbol)
        # print("bool_data:", bool_data)

        # Split bits for in-phase (real) and quadrature (imaginary) components
        half_bits = num_bits_per_symbol // 2
        real_part = bool_data[:, :half_bits]
        imag_part = bool_data[:, half_bits:]

        # print("real_indices:", real_part)
        # print("imag_indices:", imag_part)
        real_indices = np.empty(int(len(bool_data)), dtype=int)
        imag_indices = np.empty(int(len(bool_data)), dtype=int)
        for idx, bool_vector in enumerate(real_part):
            real_int = 0
            for bit in bool_vector:
                real_int = (real_int << 1) | bit
            real_indices[idx] = real_int
        
        for idx, bool_vector in enumerate(imag_part):
            imag_int = 0
            for bit in bool_vector:
                imag_int = (imag_int << 1) | bit
            imag_indices[idx] = imag_int

        # print("real_indices:", real_indices)
        # print("imag_indices:", imag_indices)

        # Map indices to amplitude levels
        i = levels[real_indices]
        q = levels[imag_indices]

        # print("real encoded:", i)
        # print("imag encoded:", q)

        modulated_data[:] =  (i + 1j * q)*self.normalization_factor

        # print(modulated_data)