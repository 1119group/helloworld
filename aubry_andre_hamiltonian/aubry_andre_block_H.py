'''
This function generates a block diagonalized Hamiltonian for a
fermionic Aubry-Andre Hamiltonian.
'''

import quantum_module as qm
import numpy as np
from scipy import io
from scipy.sparse import lil_matrix
from scipy.misc import comb
import os
import pickle


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
    bs_fname = 'cache/basisset_' + str(N) + 'spins_' + str(current_j) + 'Sz'
    bidict_fname = 'cache/basisinddict_' + str(N)
    bidict_fname += 'spins_' + str(current_j) + 'Sz'
    if os.path.isfile(bs_fname) and os.path.isfile(bidict_fname):
        data0 = open(bs_fname, 'rb')
        data1 = open(bidict_fname, 'rb')
        current_j_basis_set = pickle.load(data0)
        basis_ind_dict = pickle.load(data1)
        data0.close()
        data1.close()
    else:
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
        with open(bs_fname, 'wb') as data0:
            pickle.dump(current_j_basis_set, data0, pickle.HIGHEST_PROTOCOL)
        with open(bidict_fname, 'wb') as data1:
            pickle.dump(basis_ind_dict, data1, pickle.HIGHEST_PROTOCOL)
    return current_j_basis_set, basis_ind_dict


def iowrapper(func, fname, options):
    """
    Wrapper for I/O operations to enable loading and dumping cache to disk.
    The data in question must be a scipy sparse matrix for this function
    to work properly.
    "func" is the function to be wrapped.
    "fname" is the name of the file to be saved/loaded.
    "options" is a list of parameters to be passed on to the function.
    """
    # TODO: Move to quantum_module in due time
    if os.path.isfile(fname):
        operator = io.loadmat(fname)['i']
    else:
        operator = func(options)
        io.savemat(fname, {'i': operator}, appendmat=False)
    return operator


def blk_off_diag_ut_nocache(options):
    """
    Privides the upper half of the off_diagonal elements of one
    block of the Hamiltonian corresponding to the given total Sz.
    Returns an upper triangle sparse matrix.
    "total_Sz" is the total spin of all the basis states in the current block.
    "pos" is the i or j coordinates of the upper left element of
    the current block in the full Hamiltonian.
    This function does not save the results on disk.
    """
    [Sx, Sy, Sz, N, total_Sz, J] = options
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


def blk_off_diag_ut(Sx, Sy, Sz, N, total_Sz, J=1):
    """
    Privides the off_diagonal elements of one block of the Hamiltonian
    corresponding to the given total Sz. Provides only an upper triangular
    matrix.
    "total_Sz" is the total spin of all the basis states in the current block.
    "pos" is the i or j coordinates of the upper left element of
    the current block in the full Hamiltonian.
    """
    options = [Sx, Sy, Sz, N, total_Sz, J]
    fname = 'cache/H_block_off_diag_ut' + str(N) + 'spins_J' + str(J) + '_'
    fname += str(total_Sz) + 'Sz.mat'
    H_curr_blk_off_diag_ut = iowrapper(blk_off_diag_ut_nocache, fname, options)
    return H_curr_blk_off_diag_ut


def blk_off_diag(Sx, Sy, Sz, N, total_Sz, J=1):
    """
    Privides the off_diagonal elements of one block of the Hamiltonian
    corresponding to the given total Sz. Provides only an upper triangular
    matrix.
    "total_Sz" is the total spin of all the basis states in the current block.
    "pos" is the i or j coordinates of the upper left element of
    the current block in the full Hamiltonian.
    """
    H_curr_blk_off_diag = blk_off_diag_ut(Sx, Sy, Sz, N, total_Sz, J)
    H_curr_blk_off_diag += H_curr_blk_off_diag.transpose()
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


def blk_full(Sx, Sy, Sz, N, h, c, total_Sz, phi=0, J=1):
    """Generates block of the Hamiltonian for a specific total <Sz>."""
    H_curr_blk_diag = blk_diag(Sx, Sy, Sz, N, h, c, total_Sz, phi)
    H_curr_blk_off_diag = blk_off_diag(Sx, Sy, Sz, N, total_Sz, J)
    return H_curr_blk_diag + H_curr_blk_off_diag


def aubry_andre_H_off_diag_nocache(options):
    """
    Provides the off-diagonal elements of the Hamiltonian.
    """
    [Sx, Sy, Sz, N, J] = options
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


def aubry_andre_H_off_diag(Sx, Sy, Sz, N, J=1):
    options = [Sx, Sy, Sz, N, J]
    fname = 'cache/aubry_H_off_diag_' + str(N) + 'spins_J' + str(J) + '.mat'
    H_off_diag = iowrapper(aubry_andre_H_off_diag_nocache, fname, options)
    return H_off_diag


def aubry_andre_H_diag_nocache(options):
    """
    Provides the diagonal elements of the Hamiltonian. This version
    does not save/load cache from disk.
    """
    [Sx, Sy, Sz, N, h, c, phi] = options
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


def aubry_andre_H_diag(Sx, Sy, Sz, N, h, c, phi=0):
    """Provides the diagonal elements of the Hamiltonian."""
    options = [Sx, Sy, Sz, N, h, c, phi]
    fname = 'cache/aubry_H_diag_' + str(N) + 'spins' + '_h' + str(h)
    fname += '_c' + str(c) + '_phi' + str(phi) + '.mat'
    H_off_diag = iowrapper(aubry_andre_H_diag_nocache, fname, options)
    return H_off_diag


def aubry_andre_H(Sx, Sy, Sz, N, h, c, phi):
    """
    A function to put together the pieces of a Hamiltonian and
    generate a full Hamiltonian.
    """
    H_off_diag = aubry_andre_H_off_diag(Sx, Sy, Sz, N)
    H_diag = aubry_andre_H_diag(Sx, Sy, Sz, N, h, c, phi)
    H = H_diag + H_off_diag
    return H
