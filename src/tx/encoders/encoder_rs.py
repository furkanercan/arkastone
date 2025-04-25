from src.utils.galois_field import GF

class ReedSolomonEncoder:
    """
    Reed-Solomon Encoder using Galois Field arithmetic.
    Rules check:
    n being 2^m-1, puncture otherwise
    total info bits = k * m
    """
    def __init__(self, n, k, m=8, primitive_poly=0x11D):
        """
        Initializes the RS encoder.
        :param n: Codeword length (Total symbols = Data symbols + Parity symbols)
        :param k: Number of data symbols
        :param primitive_poly: Primitive polynomial for GF(2^m)
        """
        assert n > k, "n must be greater than k for Reed-Solomon encoding"
        self.n = n
        self.k = k
        self.gf = GF(m, primitive_poly=primitive_poly)
        self.generator_poly = self._compute_generator_poly()
    
    def _compute_generator_poly(self):
        """Computes the generator polynomial g(x) for the Reed-Solomon code.
        The generator polynomial g(x) is a fundamental part of Reed-Solomon encoding. 
        It defines how redundant parity symbols are generated to detect and correct errors.
        g(x)=(x-alpha^0)*(x-alpha^1)*(x-alpha^2)...(x-alpha^(n-k-1))
        """
        g = [1]
        for i in range(self.n - self.k):
            # print("g: ",g)
            # print("self.gf.exp_table[i]: ", self.gf.exp_table[i])
            g = self._poly_mul(g, [1, self.gf.exp_table[i]])
        return g

    def _poly_mul(self, p1, p2):
        """Multiply two polynomials in GF(2^m)."""
        if p1 is None or p2 is None:
            raise ValueError("Polynomial multiplication received None as input.")
        
        result = [0] * (len(p1) + len(p2) - 1)
        for i in range(len(p1)):
            for j in range(len(p2)):
                result[i + j] ^= self.gf.mul(p1[i], p2[j])
        
        return result
    
    def encode(self, data):
        """
        Encodes a message using Reed-Solomon encoding.
        :param data: List of k data symbols
        :return: List of n encoded symbols (data + parity)
        """
        assert len(data) == self.k, "Input data must have length k"
        msg_poly = data + [0] * (self.n - self.k)  # Append parity placeholders
        
        # Perform division msg(x) / g(x) to compute remainder (parity)
        for i in range(self.k):
            coef = msg_poly[i]
            if coef != 0: 
                for j in range(len(self.generator_poly)):
                    msg_poly[i + j] ^= self.gf.mul(self.generator_poly[j], coef)
        
        # Append parity symbols to data
        parity = msg_poly[self.k:]
        # print(parity)
        return data + parity