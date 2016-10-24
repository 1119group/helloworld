"""
Solution that is quickly hacked together. Needs to be integrated with the
rest of the code base.
"""
import quantum_module as qm
import scipy as scp
import numpy as np


def bin_to_dec(l):
    """Converts a list "l" of 1s and 0s into a decimal"""
    return int(''.join(map(str, l)), 2)


def create_complete_basis(N, current_j):
    """Creates a complete basis for the current total <Sz>"""
    dim = 2 ** N
    spin_ups = round(0.5 * N + current_j)
    spin_downs = N - spin_ups
    blksize = int(round(scp.misc.comb(N, spin_ups)))
    basis_seed = [0] * spin_downs + [1] * spin_ups
    basis = basis_seed
    # "to_diag" is a dict that maps ordinary indices to block diagonalized
    #  indices. "to_ord" is the opposite.
    basis_set, to_diag, to_ord = [], {}, {}
    for i in range(blksize):
        try:
            basis = qm.permute_one_zero(basis)
        except IndexError:                # When current_j is N // 2 or -N // 2
            pass
        basis_set.append(basis[:])
        decimal_basis = bin_to_dec(basis)
        # i is the index within only this block
        to_diag[dim - decimal_basis - 1] = i
        to_ord[i] = dim - decimal_basis - 1
    return basis_set, to_diag, to_ord


def similarity_trans_matrix(N):
    """
    Returns a matrix U such that Uv = v' with v in the tensor product
    basis arrangement and v' in the spin block basis arrangement.

    Args: "N" System size
    Returns: Sparse matrix (CSC matrix)
    """
    offset = 0
    dim = 2 ** N
    data = np.ones(dim)
    row_ind = np.empty(dim)
    col_ind = np.empty(dim)
    current_pos = 0                     # current position along the data array
    for current_j in range(N // 2, -N // 2 - 1, -1):
        spin_ups = round(0.5 * N + current_j)
        blksize = int(round(scp.misc.comb(N, spin_ups)))
        to_diag = create_complete_basis(N, current_j)[1]
        for ord_ind, diag_ind in to_diag.items():
            row_ind[current_pos] = diag_ind + offset
            col_ind[current_pos] = ord_ind
            current_pos += 1
        offset += blksize
    return scp.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(dim, dim))
