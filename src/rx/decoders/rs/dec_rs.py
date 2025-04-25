from src.utils.galois_field import GF

class ReedSolomonDecoder:
    """
    Reed-Solomon Decoder using Galois Field arithmetic.
    """
    def __init__(self, n, k, m=8, primitive_poly=0x11D):
        """
        Initializes the RS decoder.
        :param n: Codeword length (Total symbols = Data symbols + Parity symbols)
        :param k: Number of data symbols
        :param m: Power of binary extension field, 2^m
        :param primitive_poly: Primitive polynomial for GF(2^m)
        """
        assert n > k, "n must be greater than k for Reed-Solomon decoding"
        self.n = n
        self.k = k
        self.m = m
        self.t = (n - k) // 2  # Maximum correctable errors
        self.gf = GF(m, primitive_poly=primitive_poly)
    
    def compute_syndromes(self, received):
        """Compute the syndrome values from the received codeword."""
        # print("Received: ", received)
        syndromes = [0] * (2 * self.t)
        for i in range(1, 2 * self.t + 1):
            syndromes[i - 1] = 0  # Initialize in GF(256)
            for j in range(self.n):
                syndromes[i - 1] ^= self.gf.mul(received[j], self.gf.exp_table[(i * j) % (self.gf.field_size - 1)])
    
        # print(f"Computed Syndromes: {syndromes}")
        return syndromes
    
    def berlekamp_massey(self, syndromes):
        """
        Implements the Berlekamp-Massey algorithm for finding the error locator polynomial.
        Input: syndromes - List of syndromes over GF
        Output: locator polynomial coefficients [1, λ1, λ2, ..., λL]
        """
        GF = type(syndromes[0])  # assume syndromes are already in GF
        n = len(syndromes)
        C = [GF(1)]
        B = [GF(1)]
        L = 0
        m = 1
        b = GF(1)

        for n_idx in range(n):
            # Compute discrepancy
            d = syndromes[n_idx]
            for i in range(1, L + 1):
                d += C[i] * syndromes[n_idx - i]

            if d == 0:
                m += 1
            else:
                T = C.copy()
                scale = d / b
                B_shifted = [GF(0)] * m + [scale * coeff for coeff in B]
                # Extend C to fit
                if len(B_shifted) > len(C):
                    C += [GF(0)] * (len(B_shifted) - len(C))
                C = [c - b for c, b in zip(C, B_shifted)]

                if 2 * L <= n_idx:
                    L = n_idx + 1 - L
                    B = T
                    b = d
                    m = 1
                else:
                    m += 1

        return C

    def berlekamp_massey_builtin(self, syndromes):
        """
        Uses the built-in galois.berlekamp_massey to get the error locator polynomial.
        """
        import galois
        GF = type(syndromes[0])  # Grab the correct field
        syndromes_gf = GF(syndromes)
        return galois.berlekamp_massey(syndromes_gf).coeffs.tolist()

    
    # def berlekamp_massey(self, syndromes):
    #     """Find the error locator polynomial using the Berlekamp-Massey Algorithm."""
    #     n = len(syndromes)
    #     sigma = [1] + [0] * n  # Error locator polynomial (starts as 1)
    #     C = [1] + [0] * n  # Auxiliary polynomial
    #     L = 0  # Number of errors found
    #     b = 1  # Previous discrepancy

    #     for r in range(n):
    #         # Compute discrepancy Δ
    #         delta = syndromes[r]
    #         for j in range(1, L + 1):
    #             delta ^= self.gf.mul(sigma[j], syndromes[r - j])

    #         print(f"Step {r}: delta={delta}, L={L}")

    #         if delta != 0:  # Only update if there's a discrepancy
    #             T = sigma[:]  # Copy sigma
    #             coef = self.gf.div(delta, b)  # Compute correction factor
    #             for j in range(n - r):
    #                 sigma[r + j] ^= self.gf.mul(coef, C[j])

    #             if 2 * L <= r:
    #                 L = r + 1 - L
    #                 C = T  # Update auxiliary polynomial
    #                 b = delta  # Update previous discrepancy

    #     return sigma[:L + 1]  # Return the trimmed error locator polynomial
    
    def find_error_positions(self, locator_poly):
        """Find error positions using Chien Search."""
        pass  # To be implemented
    
    def compute_error_magnitudes(self, syndromes, locator_poly, error_positions):
        """Compute the error magnitudes using the Forney algorithm."""
        pass  # To be implemented
    
    def correct_errors(self, received, error_positions, error_magnitudes):
        """Correct the errors in the received codeword."""
        for i in range(len(error_positions)):
            received[error_positions[i]] ^= error_magnitudes[i]
        return received
    
    def decode(self, received):
        """Perform full RS decoding on the received codeword."""
        syndromes = self.compute_syndromes(received)
        if max(syndromes) == 0:
            return received[:self.k]  # No errors detected
        
        locator_poly = self.find_error_locator_polynomial(syndromes)
        error_positions = self.find_error_positions(locator_poly)
        error_magnitudes = self.compute_error_magnitudes(syndromes, locator_poly, error_positions)
        corrected_codeword = self.correct_errors(received, error_positions, error_magnitudes)
        
        return corrected_codeword[:self.k]  # Return only the corrected data symbols
