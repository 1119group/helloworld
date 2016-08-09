'''
This function generates a block diagonalized Hamiltonian for a
fermionic Aubry-Andre Hamiltonian.
'''

import quantum_module as qm
import numpy as np
from scipy.sparse import lil_matrix
from scipy.misc import comb


def bin2dec(l):
    '''
    Convert a list of 1's and 0's, a binary representation of a number,
    into a decimal number.
    "l" must be a list/array of 1's and 0's.
    '''
    n = len(l)
    dec = 0
    for i in range(n):
        dec += l[n - 1 - i] * 2**i
    return dec


def basis_set(N, blk_sz, j_max, current_j):
    """
    Returns a list of the full set of basis for the current total <Sz>.
    "blk_sz" is the length of the block of the Hamiltonian block that we
    are seeking the basis set for.
    "current_j" is the total <Sz> for the current block.
    """
    # Find all the binary representations of the current j.
    D = 2**N
    basis_set_seed = [0] * N
    basis_ind_dict = {}
    basis_index = 0
    for n in range(j_max + current_j):
        basis_set_seed[N - 1 - n] = 1
    current_j_basis_set = []
    if blk_sz != 1:
        for i in range(blk_sz):
            current_j_basis_set.append(qm.permute_one_zero(
                                       basis_set_seed)[:])
            # Create a dictionary that links the decimal
            #  representation of a basis and its position in this
            #  particular way of basis ordering.
            basis_ind_dict[D - 1 - bin2dec(basis_set_seed)] = basis_index
            basis_index += 1
    else:
        # The permute function cannot permute lists for which there is
        #  only one permutation.
        current_j_basis_set.append(basis_set_seed)
    return current_j_basis_set, basis_ind_dict


def blk_off_diag(Sx, Sy, Sz, N, total_Sz, J=1):
    """
    Privides the upper half of the off_diagonal elements of one
    block of the Hamiltonian corresponding to the given total Sz.
    Returns an upper triangle sparse matrix.
    "total_Sz" is the total spin of all the basis states in the current block.
    "pos" is the i or j coordinates of the upper left element of
    the current block in the full Hamiltonian.
    """
    D = Sx.get_shape()[0]**N
    # Side length of the current block.
    j_max = int(round(0.5 * N))
    blk_sz = int(round(comb(N, j_max - total_Sz)))
    current_j_basis_set, basis_ind_dict = basis_set(N, blk_sz,
                                                    j_max, total_Sz)
    H_curr_blk_off_diag = lil_matrix((blk_sz, blk_sz), dtype=complex)

    # Fill in the off-diagonal elements for the current block.
    for i in range(blk_sz):
        k = N - 1
        nz_list = []
        while k >= 0:
            curr_bs = current_j_basis_set[i][:]
            while True:
                if abs(curr_bs[k] - curr_bs[k - 1]) == 1:
                    break
                else:
                    k -= 1
            curr_bs[k], curr_bs[k - 1] = curr_bs[k - 1], curr_bs[k]
            curr_ind = basis_ind_dict[D - 1 - bin2dec(curr_bs)]
            if curr_ind > i:
                nz_list.append(curr_ind)
            k -= 1
        for k in nz_list:
            H_curr_blk_off_diag[i, k] = 0.5 * J
    return H_curr_blk_off_diag


def blk_diag(Sx, Sy, Sz, N, h, c, total_Sz, phi=0):
    """
    Provides the diagonal entries of a block of the Hamiltonian
    corresponding to the given total Sz.
    "total_Sz" is the total spin of all the basis states in the current block.
    """
    # Side length of the current block.
    j_max = int(round(0.5 * N))
    blk_sz = int(round(comb(N, int(j_max - total_Sz))))
    current_j_basis_set, basis_ind_dict = basis_set(N, blk_sz,
                                                    j_max, total_Sz)

    # Create the field
    field = np.empty(N)
    for i in range(N):
        field[i] = h * np.cos(c * 2 * np.pi * (i + 1) + phi)

    H_curr_blk_diag = lil_matrix((blk_sz, blk_sz), dtype=complex)

    # Fill in the diagonal elements for the current block.
    for i in range(blk_sz):
        h_sum = 0
        tot = 0
        for k in range(N):
            # h(i)Sz(i)
            if current_j_basis_set[i][k] == 0:
                h_sum -= field[k]
            elif current_j_basis_set[i][k] == 1:
                h_sum += field[k]

            # Sz(i)Sz(i+1)
            diff = current_j_basis_set[i][N - 1 - k]
            diff -= current_j_basis_set[i][N - 2 - k]
            tot += abs(diff)
        imb = N - 2 * tot
        H_curr_blk_diag[i, i] = imb * 0.25 + 0.5 * h_sum
    return H_curr_blk_diag


def aubry_andre_H_off_diag(Sx, Sy, Sz, N, J=1):
    """
    Provides the upper half of the off-diagonal elements of the
    Hamiltonian. Returns an upper triangle sparse matrix.
    """
    D = Sx.get_shape()[0]**N
    H_off_diag = lil_matrix((D, D), dtype=complex)
    j_max = int(round(0.5 * N))
    current_j = j_max
    pos = 0
    while current_j >= -1 * j_max:
        blk_sz = int(round(comb(N, j_max - current_j)))
        if blk_sz != 1:
            H_j_off_diag = blk_off_diag(Sx, Sy, Sz, N, current_j, J)
            H_off_diag[pos:pos + blk_sz,
                       pos:pos + blk_sz] += H_j_off_diag
        pos += blk_sz
        current_j -= 1
    return H_off_diag


def aubry_andre_H_diag(Sx, Sy, Sz, N, h, c, phi=0):
    """Provides the diagonal elements of the Hamiltonian."""
    D = 2**N
    H_diag = lil_matrix((D, D), dtype=complex)
    j_max = int(round(0.5 * N))
    current_j = j_max
    pos = 0
    while current_j >= -1 * j_max:
        blk_sz = int(round(comb(N, int(j_max - current_j))))
        H_j_diag = blk_diag(Sx, Sy, Sz, N, h, c, current_j, phi)
        H_diag[pos:pos + blk_sz,
               pos:pos + blk_sz] += H_j_diag
        pos += blk_sz
        current_j -= 1
    return H_diag
