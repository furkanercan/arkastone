import numpy as np

class GF:
    """
    Galois Field Arithmetic for GF(2^m) used in Reed-Solomon encoding and decoding.
    """
    def __init__(self, m=8, primitive_poly=0x11D):
        self.m = m
        self.primitive_poly = primitive_poly
        self.field_size = 2**m
        self.exp_table = [0] * self.field_size
        self.log_table = [0] * self.field_size
        self._generate_tables()

    def _generate_tables(self):
        """Generate exponentiation and logarithm tables for GF(2^m)."""
        alpha = 1 # Start with the base value alpha^0 = 1
        for i in range(self.field_size - 1):
            self.exp_table[i] = alpha
            self.log_table[alpha] = i
            
            # Find the next alpha value by shifting it to the left (multiply by 2)
            # In GF, this is equivalent to multiply by the base value alpha

            alpha <<= 1 #shift left once towards obtaining the next alpha
            if alpha & self.field_size: # If overflowed, normalize by primitive polynomial
                alpha ^= self.primitive_poly # Reduce modulo primitive polynomial
        
        # Recall GF is 2^m-1, and we have one extra index that cycles around (for ease)
        self.exp_table[self.field_size - 1] = self.exp_table[0]  # Wrap around for ease of indexing, fast modulus, easy allocation
    
    def add(self, a, b):
        """Addition in GF(2^m) is XOR."""
        return a ^ b

    def sub(self, a, b):
        """Subtraction in GF(2^m) is also XOR (same as addition)."""
        return a ^ b

    def mul(self, a, b):
        """Multiplication using logarithm and exponentiation tables."""
        if a == 0 or b == 0:
            return 0
        return self.exp_table[(self.log_table[a] + self.log_table[b]) % (self.field_size - 1)]

    def div(self, a, b):
        """Division using logarithm and exponentiation tables."""
        # print("a:", a)
        # print("b:", b)
        # print("self.log_table[a]:", self.log_table[a])
        # print("self.log_table[b]:", self.log_table[b])
        # print("self.field_size:", self.field_size)
        if b == 0:
            raise ZeroDivisionError("Division by zero in GF(2^m)")
        if a == 0:
            return 0
        return self.exp_table[(self.log_table[a] - self.log_table[b]) % (self.field_size - 1)]

    def inv(self, a):
        """Multiplicative inverse in GF(2^m)."""
        if a == 0:
            raise ZeroDivisionError("Inverse of zero is undefined in GF(2^m)")
        return self.exp_table[self.field_size - 1 - self.log_table[a]]   

