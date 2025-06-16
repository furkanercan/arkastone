import numpy as np

def octal_to_binary_list(octal):
#     return [int(b) for b in bin(int(str(octal), 8))[2:].zfill(7)]
    return [int(b) for b in bin(octal)[2:].zfill(7)]

# LTE generator polynomials
G0 = octal_to_binary_list(0o133)
G1 = octal_to_binary_list(0o171)
G2 = octal_to_binary_list(0o165)
generators = [G0, G1, G2]

def convolutional_encode_tail_biting_lte(u):
    """
    LTE rate-1/3 tail biting convolutional encoder.
    u: input bits (list or np.array of 0/1), must be at least 7 bits long.
    returns: encoded sequence (list of 0/1), length = 3 * len(u)
    """
    K = len(G0)  # Constraint length
    N = len(u)

    if N < K:
        raise ValueError("Input must be at least as long as the constraint length")

    # Determine the initial state by simulating the encoder on u
    shift_reg = [0] * K
    for i in range(N):
        shift_reg = [u[i]] + shift_reg[:-1]

    initial_state = shift_reg.copy()

    # Tail biting: use final state as initial state
    shift_reg = initial_state.copy()
    output = []

    for i in range(N):
        shift_reg = [u[i]] + shift_reg[:-1]
        for g in generators:
            bit = sum([g[j] & shift_reg[j] for j in range(K)]) % 2
            output.append(bit)

    return output