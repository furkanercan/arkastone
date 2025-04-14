import numpy as np
import math

class Demodulator:
    def __init__(self, config):
        self.scheme = config["type"].lower()
        self.demod_type = config["demod_type"].lower()

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

    def demodulate(self, vec_output, input_data, awgn_var=None):
        """
        Wrapper function to handle modulation or demodulation.

        Args:
            mod_scheme (str): Modulation scheme ('bpsk' or 'qpsk').
            demod_type (str): Demodulation type ('soft' or 'hard').
            vec_output (np.ndarray): Output array to store results (LLRs or bits).
            input_data (np.ndarray): Input data for modulation/demodulation.
            awgn_var (float, optional): Noise variance for soft demodulation.

        Returns:
            None: Updates `vec_output` in place.
        """
        if self.scheme == "bpsk":
            if self.demod_type == "soft":
                if awgn_var is None:
                    raise ValueError("Noise variance is required for soft demodulation.")
                self.softDemod_bpsk(vec_output, input_data, awgn_var)
            elif self.demod_type == "hard":
                self.hardDemod_bpsk(vec_output, input_data)
            else:
                raise ValueError(f"Unsupported demodulation type: {self.demod_type}")
        elif self.scheme == "qpsk":
            if self.demod_type == "soft":
                if awgn_var is None:
                    raise ValueError("Noise variance is required for soft demodulation.")
                self.softDemod_qpsk(vec_output, input_data, awgn_var)
            elif self.demod_type == "hard":
                self.hardDemod_qpsk(vec_output, input_data)
            else:
                raise ValueError(f"Unsupported demodulation type: {self.demod_type}")
        elif self.scheme == "16qam":
            if self.demod_type == "soft":
                if awgn_var is None:
                    raise ValueError("Noise variance is required for soft demodulation.")
                self.softDemod_mqam(vec_output, input_data, awgn_var, 16)
            elif self.demod_type == "hard":
                self.hardDemod_mqam(vec_output, input_data, 16)
            else:
                raise ValueError(f"Unsupported demodulation type: {self.demod_type}")
        elif self.scheme == "64qam":
            if self.demod_type == "soft":
                if awgn_var is None:
                    raise ValueError("Noise variance is required for soft demodulation.")
                self.softDemod_mqam(vec_output, input_data, awgn_var, 64)
            elif self.demod_type == "hard":
                self.hardDemod_mqam(vec_output, input_data, 64)
            else:
                raise ValueError(f"Unsupported demodulation type: {self.demod_type}")
        elif self.scheme == "256qam":
            if self.demod_type == "soft":
                if awgn_var is None:
                    raise ValueError("Noise variance is required for soft demodulation.")
                self.softDemod_mqam(vec_output, input_data, awgn_var, 256)
            elif self.demod_type == "hard":
                self.hardDemod_mqam(vec_output, input_data, 256)
            else:
                raise ValueError(f"Unsupported demodulation type: {self.demod_type}")
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.scheme}")


    def hardDemod_bpsk(self, vec_hd, input_data):
        input_data = np.array(input_data)
        vec_hd[:] = np.where(input_data < 0, 1, 0)

    def softDemod_bpsk(self, vec_llr, input_data, awgn_var):
        vec_llr[:] = 2 * (input_data) / awgn_var 



        
    def hardDemod_qpsk(self, vec_hd, input_data):
        """
        QPSK demodulation: Recover bits from QPSK symbols.

        Args:
            input_data (np.ndarray): Complex QPSK symbols.

        Returns:
            np.ndarray: Recovered bit sequence.
        """
        i = np.real(input_data)  # In-phase component
        q = np.imag(input_data)  # Quadrature component

        # Map signs to bits
        bits_i = (i < 0).astype(int)
        bits_q = (q < 0).astype(int)

        # Combine bits and flatten
        vec_hd[:] = np.column_stack((bits_i, bits_q)).flatten()
    

    def softDemod_qpsk(self, vec_llr, input_data, awgn_var):
        """
        Perform QPSK soft demodulation to compute LLRs.

        Args:
            input_data (np.ndarray): Complex QPSK symbols (received).
            awgn_var (float): Noise variance (sigma^2).

        Returns:
            np.ndarray: LLRs for each bit (1D array, 2 LLRs per symbol).
        """
        # Compute scaling factor for LLRs
        scaling_factor = 2 / awgn_var

        # Extract real and imaginary parts
        llr_i = scaling_factor * np.real(input_data)  # LLR for in-phase bits
        llr_q = scaling_factor * np.imag(input_data)  # LLR for quadrature bits

        # Interleave LLRs: [LLR_i1, LLR_q1, LLR_i2, LLR_q2, ...]
        # llrs = np.empty(2 * len(input_data))
        vec_llr[0::2] = llr_i  # Place LLRs for in-phase bits in even indices
        vec_llr[1::2] = llr_q  # Place LLRs for quadrature bits in odd indices


    def hardDemod_mqam(self, vec_hd, input_data, m):
        """
        Perform hard demodulation for M-QAM.

        Args:
            input_data (np.ndarray): Received symbols (complex values).
            m (int): Modulation order (e.g., 16 for 16-QAM, 64 for 64-QAM).

        Returns:
            np.ndarray: Demodulated bit sequence.
        """

        demod_map = np.array([
            [0, 0],  # Mapping for 0
            [0, 1],  # Mapping for 1
            [1, 0],  # Mapping for 2
            [1, 1]   # Mapping for 3
        ])


        if np.log2(m) % 1 != 0:
            raise ValueError("Modulation order M must be a power of 2.")
        
        num_bits_per_symbol = int(np.log2(m))
        # levels = np.arange(np.sqrt(m) - 1, -np.sqrt(m), step=-2)
        levels = np.array([-3, -1, 3, 1], dtype=int)

        # Quantize real and imaginary parts to nearest constellation points
        real_indices = np.empty(len(input_data), dtype=int)
        imag_indices = np.empty(len(input_data), dtype=int)
        for idx, value in enumerate(np.real(input_data)):
            if(value < -2):
                real_indices[idx] = 0
            elif(value < 0):
                real_indices[idx] = 1
            elif(value < 2):
                real_indices[idx] = 3
            else:
                real_indices[idx] = 2
        
        for idx, value in enumerate(np.imag(input_data)):
            if(value < -2):
                imag_indices[idx] = 0
            elif(value < 0):
                imag_indices[idx] = 1
            elif(value < 2):
                imag_indices[idx] = 3
            else:
                imag_indices[idx] = 2
        

        modulated_int_data = np.ravel(np.column_stack((real_indices, imag_indices)))

        vec_hd[:] = demod_map[modulated_int_data].flatten()

        # print(vec_hd.flatten())


        # print(input_data)
        # print(real_indices)
        # print(imag_indices)
        # print(np.ravel(np.column_stack((real_indices, imag_indices))))
        # print(np.vstack((real_indices, imag_indices)))

        # # Quantize real and imaginary parts to nearest constellation points
        # real_indices = np.digitize(np.real(input_data), levels) - 1
        # imag_indices = np.digitize(np.imag(input_data), levels) - 1

        # # Convert indices back to bits
        # half_bits = num_bits_per_symbol // 2
        # real_bits = np.unpackbits(real_indices.astype(np.uint8), bitorder="little")[-half_bits:]
        # imag_bits = np.unpackbits(imag_indices.astype(np.uint8), bitorder="little")[-half_bits:]
        
        # # Combine real and imaginary bits
        # vec_hd[:] = np.hstack((real_bits, imag_bits)).flatten()


    def softDemod_mqam(self, vec_llr, input_data, awgn_var, m):
        """
        Perform soft demodulation for M-QAM to compute LLRs.

        Args:
            input_data (np.ndarray): Received symbols (complex).
            m (int): Modulation order (e.g., 16 for 16-QAM, 64 for 64-QAM).
            awgn_var (float): Noise variance (sigma^2).

        Returns:
            np.ndarray: LLRs for each bit (length = log2(M) * number of symbols).
        """
        if np.log2(m) % 1 != 0:
            raise ValueError("Modulation order M must be a power of 2.")
        
        num_bits_per_symbol = int(np.log2(m))
        scaling_factor = 2 / awgn_var

        # print("input_data:", input_data)

        # # Gray-coded amplitude levels
        # # levels = np.arange(np.sqrt(m) - 1, -np.sqrt(m), step=-2)
        # # print(levels)
        
        # def exact_llr_b1(z, sigma2):
        #     num = np.exp(-(np.real(z) + 3)**2 / (2 * sigma2)) + np.exp(-(np.real(z) + 1)**2 / (2 * sigma2))
        #     den = np.exp(-(np.real(z) - 3)**2 / (2 * sigma2)) + np.exp(-(np.real(z) - 1)**2 / (2 * sigma2))
        #     return np.log(num / den)

        # def exact_llr_b2(z, sigma2):
        #     num = np.exp(-(np.real(z) + 3)**2 / (2 * sigma2)) + np.exp(-(np.real(z) - 3)**2 / (2 * sigma2))
        #     den = np.exp(-(np.real(z) - 1)**2 / (2 * sigma2)) + np.exp(-(np.real(z) + 1)**2 / (2 * sigma2))
        #     return np.log(num / den) 

        # def exact_llr_b3(z, sigma2):
        #     num = np.exp(-(np.imag(z) + 3)**2 / (2 * sigma2)) + np.exp(-(np.imag(z) + 1)**2 / (2 * sigma2))
        #     den = np.exp(-(np.imag(z) - 3)**2 / (2 * sigma2)) + np.exp(-(np.imag(z) - 1)**2 / (2 * sigma2))
        #     return np.log(num / den)

        # def exact_llr_b4(z, sigma2):
        #     num = np.exp(-(np.imag(z) + 3)**2 / (2 * sigma2)) + np.exp(-(np.imag(z) - 3)**2 / (2 * sigma2))
        #     den = np.exp(-(np.imag(z) - 1)**2 / (2 * sigma2)) + np.exp(-(np.imag(z) + 1)**2 / (2 * sigma2))
        #     return np.log(num / den)        
        
        def approx_llr_b1(z, sigma2):
            num = -np.minimum((np.real(z) + 3*self.normalization_factor)**2,(np.real(z) + 1*self.normalization_factor)**2)
            den = -np.minimum((np.real(z) - 3*self.normalization_factor)**2, (np.real(z) - 1*self.normalization_factor)**2)
            return (num-den)/(2*sigma2)

        def approx_llr_b2(z, sigma2):
            num = -np.minimum((np.real(z) + 3*self.normalization_factor)**2,(np.real(z) - 3*self.normalization_factor)**2)
            den = -np.minimum((np.real(z) + 1*self.normalization_factor)**2, (np.real(z) - 1*self.normalization_factor)**2)
            return (num-den)/(2*sigma2)
        
        def approx_llr_b3(z, sigma2):
            num = -np.minimum((np.imag(z) + 3*self.normalization_factor)**2,(np.imag(z) + 1*self.normalization_factor)**2)
            den = -np.minimum((np.imag(z) - 3*self.normalization_factor)**2, (np.imag(z) - 1*self.normalization_factor)**2)
            return (num-den)/(2*sigma2)

        def approx_llr_b4(z, sigma2):
            num = -np.minimum((np.imag(z) + 3*self.normalization_factor)**2,(np.imag(z) - 3*self.normalization_factor)**2)
            den = -np.minimum((np.imag(z) + 1*self.normalization_factor)**2, (np.imag(z) - 1*self.normalization_factor)**2)
            return (num-den)/(2*sigma2)
        

        for idx, val in enumerate(input_data):
            vec_llr[4*idx + 0] = approx_llr_b1(val,awgn_var)
            vec_llr[4*idx + 1] = approx_llr_b2(val,awgn_var)
            vec_llr[4*idx + 2] = approx_llr_b3(val,awgn_var)
            vec_llr[4*idx + 3] = approx_llr_b4(val,awgn_var)

        # print("vec_llr:", vec_llr)