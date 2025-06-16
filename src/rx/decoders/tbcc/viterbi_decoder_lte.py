import numpy as np

class ViterbiDecoder:
    def __init__(self, K=7, generators=(0o133, 0o171, 0o165)):
        self.K = K
        self.num_states = 2 ** (K - 1)
        self.generators = generators
        self.trellis = self._build_trellis()

    def _build_trellis(self):
        trellis = {}
        for state in range(self.num_states):
            trellis[state] = {}
            for bit in (0, 1):
                next_state = ((state << 1) | bit) & (self.num_states - 1)
                output = [self._encode_bit(state, bit, g) for g in self.generators]
                trellis[state][bit] = (next_state, output)
        return trellis

    def _encode_bit(self, state, bit, generator):
        reg = (state << 1) | bit
        val = reg & generator
        return bin(val).count("1") % 2

    def decode(self, received, tb_length=None):
        n = len(received) // len(self.generators)
        received = np.array(received).reshape((n, len(self.generators)))
        path_metrics = np.full((n + 1, self.num_states), np.inf)
        path_metrics[0][0] = 0  # known start state: 0
        paths = np.full((n, self.num_states), -1, dtype=int)

        for t in range(n):
            for state in range(self.num_states):
                if path_metrics[t][state] == np.inf:
                    continue
                for bit in (0, 1):
                    next_state, expected = self.trellis[state][bit]
                    dist = np.sum(received[t] != expected)
                    new_metric = path_metrics[t][state] + dist
                    if new_metric < path_metrics[t + 1][next_state]:
                        path_metrics[t + 1][next_state] = new_metric
                        paths[t][next_state] = state | (bit << self.K)  # encode bit in higher bits

        # Traceback
        decoded_bits = []
        state = np.argmin(path_metrics[n])  # assume best end state
        for t in reversed(range(n)):
            prev = paths[t][state]
            bit = (prev >> self.K) & 1
            decoded_bits.append(bit)
            state = prev & (self.num_states - 1)

        return decoded_bits[::-1]  # reverse to get correct order
