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
        dec += l[n-1-i]*2**i
    return dec


def basis_set(N, block_size, j_max, current_j):
    """Returns a list of the full set of basis for the current total <Sz>"""
    # Find all the binary representations of the current j.
    basis_set_seed = [0]*N
    basis_ind_dict = {}
    basis_index = 0
    for n in range(j_max+current_j):
        basis_set_seed[N-1-n] = 1
    current_j_basis_set = []
    if block_size != 1:
        for i in range(block_size):
            current_j_basis_set.append(qm.permute_one_zero(
                                       basis_set_seed)[:])
            # Create a dictionary that links the decimal
            #  representation of a basis and its position in this
            #  particular way of basis ordering.
            basis_ind_dict[bin2dec(basis_set_seed)] = basis_index
            basis_index += 1
    else:
        # The permute function cannot permute lists for which there is
        #  only one permutation.
        current_j_basis_set.append(basis_set_seed)
    return current_j_basis_set, basis_ind_dict


def aubry_andre_H_off_diag(Sx, Sy, Sz, N, J=1):
    """
    Provides the upper half of the off-diagonal elements of the
    Hamiltonian.
    """
    D = Sx.get_shape()[0]**N
    H_off_diag = lil_matrix((D, D), dtype=complex)
    j_max = int(round(0.5*N))
    current_j = j_max
    curr_pos = 0
    while current_j >= -1*j_max:
        # Side length of the current block.
        block_size = int(round(comb(N, j_max-current_j)))
        current_j_basis_set, basis_ind_dict = basis_set(N, block_size,
                                                        j_max, current_j)

        # Fill in the off-diagonal elements for the current block.
        if block_size != 1:
            for i in range(block_size):
                k = N-1
                nz_list = []
                while k >= 0:
                    curr_bs = current_j_basis_set[i][:]
                    while True:
                        if abs(curr_bs[k]-curr_bs[k-1]) == 1:
                            break
                        else:
                            k -= 1
                    curr_bs[k], curr_bs[k-1] = curr_bs[k-1], curr_bs[k]
                    curr_ind = basis_ind_dict[bin2dec(curr_bs)]
                    if curr_ind > i:
                        nz_list.append(curr_ind)
                    k -= 1
                for k in nz_list:
                    H_off_diag[curr_pos+i, curr_pos+k] = 0.5*J
                    # H_off_diag[curr_pos+k,curr_pos+i] = 0.5*J

        curr_pos += block_size
        current_j -= 1
    return H_off_diag


def aubry_andre_H_diag(Sx, Sy, Sz, N, h, c, phi=0):
    """Provides the diagonal elements of the Hamiltonian."""
    field = np.empty(N)
    for i in range(N):
        field[i] = h*np.cos(c*2*np.pi*i+phi)

    D = Sx.get_shape()[0]**N
    H_diag = lil_matrix((D, D), dtype=complex)
    j_max = int(round(0.5*N))
    current_j = j_max
    curr_pos = 0
    while current_j >= -1*j_max:
        # Side length of the current block.
        block_size = int(round(comb(N, j_max-current_j)))
        current_j_basis_set, basis_ind_dict = basis_set(N, block_size,
                                                        j_max, current_j)

        # Fill in the diagonal elements for the current block.
        for i in range(block_size):
            h_sum = 0
            tot = 0
            for k in range(N):
                # h(i)Sz(i)
                if current_j_basis_set[i][k] == 0:
                    h_sum -= field[k]
                elif current_j_basis_set[i][k] == 1:
                    h_sum += field[k]

                # Sz(i)Sz(i+1)
                diff = current_j_basis_set[i][N-1-k]
                diff -= current_j_basis_set[i][N-2-k]
                tot += abs(diff)
            imb = N-2*tot
            H_diag[curr_pos+i, curr_pos+i] = 1*imb*0.25 - 0.5*h_sum

        curr_pos += block_size
        current_j -= 1
    return H_diag
