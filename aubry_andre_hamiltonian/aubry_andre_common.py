"""
This module provides common function for many calculations in dealing with
block diagonalized Hamiltonians.

8-22-2016
"""
# TODO: Update/complete all the doc-strings to describe in depth the
#       arguments and returns.
# TODO: Add more comments.

import quantum_module as qm
from aubry_andre_H import aubry_andre_H
import aubry_andre_block_H as aubryH
import numpy as np
from scipy.sparse import dok_matrix, lil_matrix, issparse
from scipy.sparse.linalg import eigsh
from scipy.misc import comb
import time


def spin2z(D, N, psi):
    """
    Rewrite a given state psi corresponding to a zero total <Sz> from
    the spin basis (the basis of the block diagonalized Hamiltonian)
    to the conventional Sz product basis. It takes a sparse vector/state
    only. Compatible with both column and row vectors.

    Args: "D" is the side length of the full Hamiltonian. It is usually
          2**N.
          "N" is the size of the particle system.
          "psi" is the vector/state we are performing this operation on.
    Returns: "psi_tz" is the vector/state rewritten in the Hamiltonian
             spin basis.
    """
    # TODO: Test the function with both a column vector and a row vector.

    vdim = psi.get_shape()[0]
    # Convert the vector into a column if it is not one already.
    if vdim == 1:
        psi = psi.transpose().conjugate()

    psi_tz = lil_matrix((D, 1), dtype=complex)
    j_max = int(round(0.5 * N))
    blk_sz = int(round(comb(N, j_max)))
    basis_set_0, basis_dict_0 = aubryH.basis_set(N, blk_sz, j_max, 0)
    shift = int(round(0.5 * (D - blk_sz)))
    for i in psi.nonzero()[0]:
        if i >= shift and i < D - shift:
            l = basis_set_0[i - shift]
            dec = aubryH.bin2dec(l)
            psi_tz[D - 1 - dec, 0] = psi[i, 0]

    # Convert the longer version of the vector back to a row vector if
    #  if was one to begin with.
    if vdim == 1:
        psi_tz = psi_tz.transpose().conjugate()
    return psi_tz


def spin2z_nouveau(dummy):
    # TODO: Generalize spin2z to work with psi's corresponding to an arbitrary
    #       total <Sz>
    return dummy


def spin2z_complete(dummy):
    # TODO: Wrapper function for spin2z_nouveau to work with any vector in
    #       the block Hamiltonian spin basis and not just those with only
    #       elements corresponding to a specific total z direction spin.
    return dummy


def spin2z_sqm(N, S):
    """
    Much like the spin2z function, this function transforms an operator
    written in the Hamiltonian spin basis into the conventional Sz product
    state basis. Only works with operators corresponding to a zero total <Sz>.

    Args: "N" is the the system size.
          "S" is the operator. It must be a square sparse matrix.
    Returns: An operator in the new basis. It will be a square sparse matrix.
    """
    D = 2**N
    S_tz = lil_matrix((D, D), dtype=complex)
    j_max = int(round(0.5 * N))
    for current_j in range(j_max, -1 * j_max - 1, -1):
        blk_sz = int(round(comb(N, j_max + current_j)))
        basis_set_0, basis_dict_0 = aubryH.basis_set(N, blk_sz, j_max,
                                                     current_j)
        shift = int(round(0.5 * (D - blk_sz)))
        for i in range(np.shape(S.nonzero()[0])[0]):
            i0 = S.nonzero()[0][i]
            i1 = S.nonzero()[1][i]
            l0 = basis_set_0[i0 - shift]
            l1 = basis_set_0[i1 - shift]
            dec0 = aubryH.bin2dec(l0)
            dec1 = aubryH.bin2dec(l1)
            S_tz[D - 1 - dec0, D - 1 - dec1] = S[i0, i1]
    return S_tz


def Sz2spin_basis(N, S):
    """
    This function takes in a Pauli z spin operator "S" in the z product state
    basis and rewrites it into the block Hamiltonian spin basis.
    It **only** works on Pauli z spin operators in the z product state!
    (I was lying--it actually works with all diagonal matrices.)

    Args: "N" is the size of the system.
          "S" is the Pauli z spin operator in the z product state basis.
    Returns: The operator in the new basis and it will be a square
             sparse matrix.
    """
    D = 2**N
    S_ts = lil_matrix((D, D), dtype=complex)
    j_max = int(round(0.5 * N))
    S_ts[0, 0] = S[0, 0]
    S_ts[-1, -1] = S[-1, -1]
    curr_pos = 1
    for current_j in range(j_max - 1, -1 * j_max, -1):
        blk_sz = int(round(comb(N, j_max + current_j)))
        basis_set_0, basis_dict_0 = aubryH.basis_set(N, blk_sz, j_max,
                                                     current_j)
        for i in basis_dict_0:
            S_ts[curr_pos + basis_dict_0[i],
                 curr_pos + basis_dict_0[i]] = S[i, i]
        curr_pos += blk_sz
    return S_ts


def recast(N, psi_short):
    """
    This function takes in a state psi which contains only the elements
    corresponding to a zero total <Sz> and augment it into one that
    would contain elements of all other total <Sz>'s.

    This by no means changes the basis into a Sz product basis. For that
    function please refer to spin2z.

    Args: "N" is the size of the particle system
          "psi_short" is a state psi which contains only the elements
          corresponding to a zero total <Sz> and augment it into one that
          would contain elements of all other total <Sz>'s.
          "psi_short" could be sparse or dense and could be a column or
          row vector. However, it must be a two-dimensional "vector" in a
          sense that, if it is a numpy array, it must have a shape of
          (1, k) or (k, 1) instead of (k, ).
    Returns: A column/row, sparse/dense vector, depending on the type and
             shape of "psi_short." For example, a dense row psi_short input
             will give you a dense row vector output.
    """
    D = 2 ** N

    # Provides compatibility to both sparse and dense vectors.
    if issparse(psi_short):
        vdim = psi_short.get_shape()[0]
        # Convert the vector into a column if it is not one already.
        if vdim == 1:
            psi_short = psi_short.transpose().conjugate()
        psi_long = lil_matrix((D, 1), dtype=complex)
    else:
        vdim = np.shape(psi_short)[0]
        # Convert the vector into a column if it not one already.
        if vdim == 1:
            psi_short = psi_short.T.conjugate()
        psi_long = np.zeros([D, 1], dtype=complex)

    j_max = int(round(0.5 * N))
    blk_sz = int(round(comb(N, j_max)))
    shift = int(round(0.5 * (D - blk_sz)))
    psi_long[shift:D - shift, :] = psi_short[:,:]

    # Convert the longer version of the vector back to a row vector if
    #  if was one to begin with.
    if issparse(psi_short):
        if vdim == 1:
            psi_long = psi_long.transpose().conjugate()
    else:
        if vdim == 1:
            psi_long = psi_long.T.conjugate()

    return psi_long


def recast_nouveau(N, psi_short, total_Sz=0):
    """
    This function takes in a state psi which contains only the elements
    corresponding to an arbitrary total <Sz> and augment it into one that
    would contain elements of all other total <Sz>'s.

    This by no means changes the basis into a Sz product basis. For that
    function please refer to spin2z.

    Args: "N" is the size of the particle system
          "psi_short" is a state psi which contains only the elements
          corresponding to a zero total <Sz> and augment it into one that
          would contain elements of all other total <Sz>'s.
          "psi_short" could be sparse or dense and could be a column or
          row vector. However, it must be a two-dimensional "vector" in a
          sense that, if it is a numpy array, it must have a shape of
          (1, k) or (k, 1) instead of (k, ).
          "total_Sz" is the total <Sz> the input vector is corresponding to.
          It is defaulted to 0.
    Returns: A column/row, sparse/dense vector, depending on the type and
             shape of "psi_short." For example, a dense row psi_short input
             will give you a dense row vector output.
    """
    # TODO: Needs to be tested. After testing, replace recast with this
    #       function. Backwards compatible.

    D = 2 ** N

    # Provides compatibility to both sparse and dense vectors.
    if issparse(psi_short):
        vdim = psi_short.get_shape()[0]
        # Convert the vector into a column if it is not one already.
        if vdim == 1:
            psi_short = psi_short.transpose().conjugate()
        psi_long = lil_matrix((D, 1), dtype=complex)
    else:
        vdim = np.shape(psi_short)[0]
        # Convert the vector into a column if it not one already.
        if vdim == 1:
            psi_short = psi_short.T.conjugate()
        psi_long = np.zeros([D, 1], dtype=complex)

    j_max = int(round(0.5 * N))
    blk_sz = int(round(comb(N, j_max - total_Sz)))
    shift = int(round(0.5 * (D - blk_sz)))
    psi_long[shift:shift + blk_sz, :] = psi_short[:, :]

    # Convert the longer version of the vector back to a row vector if
    #  if was one to begin with.
    if issparse(psi_short):
        if vdim == 1:
            psi_long = psi_long.transpose().conjugate()
    else:
        if vdim == 1:
            psi_long = psi_long.T.conjugate()

    return psi_long


def energy_density(psi, H):
    """
    This function calculates the energy density (<ψ|H|ψ> - E_0)/(E_max - E_0).

    Args: "psi" is the state which energy density will be calculated.
          "H" is the Hamiltonian in question.
    Returns: 1. Energy density. A float.
             2. Expectation value of |ψ> and H. A float.
    """
    E_max = eigsh(H, k=1, which='LA', maxiter=1e6, return_eigenvectors=False)
    E_min = eigsh(H, k=1, which='SA', maxiter=1e6, return_eigenvectors=False)
    E = np.append(E_min, E_max)
    ev = qm.exp_value(H, psi)
    e = (ev - E[0]) / (E[-1] - E[0])
    return e, ev


def z_basis(N, dummy):
    """Returns a state psi in the Sz product state basis."""
    dim = 2
    D = dim**N
    s = 0
    for k in range(N):
        s += int(index[k] * 2**k)
    psi_0 = dok_matrix((D, 1), dtype=complex)
    psi_0[s, 0] = 1
    return psi_0


def spin_basis(N, counter):
    """Returns a state psi in the spin basis."""
    D = 2**N
    zero_Sz_basis_count = int(round(comb(N, 0.5 * N)))
    psi_0 = dok_matrix((D, 1), dtype=complex)
    blk_pos = int(round(0.5 * (D - zero_Sz_basis_count)))
    psi_0[blk_pos + counter, 0] = 1
    return psi_0


def spin_basis_0(N, counter):
    """
    Returns a state psi in the spin basis. The vector only contains
    elements corresponding to a zero total <Sz>.
    """
    zero_Sz_basis_count = int(round(comb(N, 0.5 * N)))
    psi_0 = dok_matrix((zero_Sz_basis_count, 1), dtype=complex)
    psi_0[counter, 0] = 1
    return psi_0


def init_state(N, H, basis):
    """
    Finds the initial state psi for a given Hamiltonian.
    "basis" is a function.

    Args: "N" is the size of the particle system.
          "H" is the Hamiltonian.
          "basis" is the function that generates psi_0.
    Returns: 1. Hamiltonian that was just passed onto the function. Kept to
                maintain code compatibility with some older functions that
                depended on this output. A sparse CSC matrix.
             2. Initial state psi that is selected. A column sparse CSC
                vector (matrix).
             3. The exit status of the function. If it returns "False" that
                means everything went as expected and a useful state is
                generated. If it returns "True" then no useful state is
                generated. A boolean.
    """
    global index

    zero_Sz_basis_count = int(round(comb(N, 0.5 * N)))
    # Create initial state psi with magnetization of 0. Here we first form
    #  a binary number which has an equal number of 1's and
    #  0's.
    index = np.zeros(N)
    for k in range(int(round(0.5 * N))):
        index[k] = 1

    error = False
    counter = 0
    while True:
        index = qm.permute_one_zero(index)
        psi_0 = basis(N, counter)
        # Make sure psi's energy density is very close to 0.5.
        e, ev = energy_density(psi_0, H)
        if abs(e - 0.5) < 0.001:
            break

        counter += 1
        # Display an error message when no suitable state is found.
        if counter >= zero_Sz_basis_count:
            error = True
            break
    return H, psi_0, error


def get_state(Sx, Sy, Sz, N, h, c, phi=0, J=1):
    """Works for the Sz product basis Hamiltonian."""
    H = aubry_andre_H(Sx, Sy, Sz, N, h, c, phi)
    H, psi_0, error = init_state(N, H, z_basis)
    return H, psi_0, error


def get_state_blk_full(H, N):
    """
    "H" is the full block diagonalized operator. Works for the spin_basis
    Hamiltonian.
    """
    H, psi, error = init_state(N, H, spin_basis)
    return psi, error


def get_state_blk(H, N):
    """
    "H" is only the center block of the block diagonalized Hamiltonian.
    Works for the spin_basis Hamiltonian.
    """
    H, psi, error = init_state(N, H, spin_basis_0)
    return psi, error


def average_adj_gap_ratio(sorted_eigenvalue_list):
    """
    Takes a list of eigenvalues that have been sorted low to high, finds the
    adjusted gap ratio for each set of 3 adjacent eigenvalues and then finds
    the average adjusted gap ratio.
    :param sorted_eigenvalue_list:
    :return adj_gap_ratio:
    """
    adj_gap_ratio = 0
    delta_n_list = np.diff(sorted_eigenvalue_list)
    for x in range(0, len(delta_n_list) - 1):
        adj_gap_ratio += min(delta_n_list[x], delta_n_list[x + 1]) /\
            max(delta_n_list[x], delta_n_list[x + 1])
    adj_gap_ratio /= len(delta_n_list) - 1
    return adj_gap_ratio


def average_vn_entropy(list_of_states, spin, N):
    """
    Take a list of states, find the Von Neumann entropy for each state and then
    find the average entropy for the set.
    :param list_of_states:
    :return avg_vn_entropy:
    """
    avg_vn_entropy = 0
    for psi in list_of_states:
        avg_vn_entropy += qm.get_vn_entropy(psi, spin, N, mode='eqsplit')
    avg_vn_entropy /= len(list_of_states)
    return avg_vn_entropy


def gen_eigenpairs(N, H, num_psis):
    """
    Generate Eigenpairs using Shift Inversion Method
    :param N:
    :param H:
    :param num_psis:
    :return:
    """
    global index
    D = 2 ** N
    E_max = eigsh(H, k=1, which='LA', maxiter=1e6, return_eigenvectors=False)
    E_min = eigsh(H, k=1, which='SA', maxiter=1e6, return_eigenvectors=False)
    E = np.append(E_min, E_max)
    target_E = .5 * (E[0] + E[1])
    psilist = []
    evals, evecs = eigsh(H, k=int(num_psis), sigma=target_E)
    evals.sort()
    evecs = np.matrix(evecs, dtype=complex)
    for i in range(evecs.shape[1]):
        psi = spin2z(D, N, lil_matrix(recast(N, evecs[:, i]), dtype=complex))
        psilist.append(psi)
    return H, psilist, evals


