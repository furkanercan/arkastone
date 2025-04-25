import numpy as np

class SimulationMemory:
    def __init__(self, len_k, len_n):
        self.info_data = np.empty(len_k, dtype=np.int32)
        self.encoded_data = np.empty(len_n, dtype=np.float32)
        self.modulated_data = np.empty(len_n, dtype=np.float32)
        self.received_data = np.empty(len_n, dtype=np.float32)
        self.decoded_data = np.empty(len_k, dtype=np.int32)

    def reset_info_data(self):
        self.info_data[:] = np.random.randint(0, 2, size=self.info_data.shape)