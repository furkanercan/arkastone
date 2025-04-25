import numpy as np
import math
from numba import njit

class UncodedDecoder():
    def __init__(self):
        pass

    def decode_chain(self, decoded_data, received_data):
        decoded_data[:] = [1 if val < 0 else 0 for val in received_data]
    
    def initialize_decoder(self):
        pass