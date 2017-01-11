import block_diag as bd
import quantum_module as qm
import numpy as np
from scipy.misc import comb


def aubry_andre_H(N, h, c, phi, J=1):
    """Open boundary conditions"""
    Sx, Sy, Sz = qm.init(0.5)
    full_S = [[qm.get_full_matrix(S, k, N) for k in range(N)] for S in [Sx, Sy, Sz]]
    inter_terms = J * sum(full_S[i][j] * full_S[i][j + 1] for j in range(N - 1)
                          for i in range(3))
    field = h * np.cos(2 * np.pi * c * np.arange(1, N + 1) + phi)
    field_terms = sum(field * full_S[2])
    H = inter_terms + field_terms
    return H.real


def block_diagonalized_H(N, h, c, phi, J=1):
    U = bd.similarity_trans_matrix(N)
    H = aubry_andre_H(N, h, c, phi, J)
    return U * H * U.T


def spin_block(N, h, c, phi, curr_j=0, J=1):
    offset = sum(comb(N, j, exact=True) for j in np.arange(0.5 * N - curr_j))
    blk_size = comb(N, round(0.5 * N + curr_j), exact=True)
    H = block_diagonalized_H(N, h, c, phi, J)
    return H[offset:offset + blk_size, offset:offset + blk_size]
