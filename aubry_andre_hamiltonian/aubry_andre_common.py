import quantum_module as qm
from aubry_andre_H import aubry_andre_H
import aubry_andre_block_H as aubryH
import numpy as np
from scipy.sparse import dok_matrix, lil_matrix, issparse
from scipy.sparse.linalg import eigsh
from scipy.misc import comb


def spin2z(D, N, psi):
    """
    Rewrite a given state psi from the spin basis (the basis of
    the block diagonalized Hamiltonian) to the conventional
    Sz product basis.
    Much faster than the other verions.
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


def spin2z_sqm(N, S):
    """
    Much like the spin2z function, this function transforms an operator
    written written in the Hamiltonian spin basis into the conventional
    Sz product state basis.
    "N" is the the system size.
    "S" is the operator. It must be a square sparse matrix.
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
    "N" is the size of the system.
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


def generate_H(Sx, Sy, Sz, N, h, c, phi):
    """
    A function to put together the pieces of a Hamiltonian and
    generate a full Hamiltonian.
    """
    H_off_diag = aubryH.aubry_andre_H_off_diag(Sx, Sy, Sz, N)
    H_diag = aubryH.aubry_andre_H_diag(Sx, Sy, Sz, N, h, c, phi)
    H = H_diag + H_off_diag
    return H


def recast(N, psi_short):
    """
    This function takes in a state psi which contains only the elements
    corresponding to a zero total <Sz> and augment it into one that
    would contain elements of all other total <Sz>'s.
    "psi_short" could be sparse or dense and could be a column or row vector.
    However, it must be a two-dimensional "vector" in a sense that, if it
    is a numpy array, it must have a shape of (1, k) or (k, 1) instead of
    (k, ).
    This by no means changes the basis into a Sz product basis. For that
    function please refer to spin2z.
    """
    D = 2 ** N

    # Provides compatability to both sparse and dense vectors.
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
    psi_long[shift:D - shift, :] = psi_short[:, :]

    # Convert the longer version of the vector back to a row vector if
    #  if was one to begin with.
    if issparse(psi_short):
        if vdim == 1:
            psi_long = psi_long.transpose().conjugate()
    else:
        if vdim == 1:
            psi_long = psi_long.T.conjugate()

    return psi_long


def energy_density(psi, H, E):
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
    """
    global index
    E_max = eigsh(H, k=1, which='LA', maxiter=1e6, return_eigenvectors=False)
    E_min = eigsh(H, k=1, which='SA', maxiter=1e6, return_eigenvectors=False)
    E = np.append(E_min, E_max)

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
        e, ev = energy_density(psi_0, H, E)
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

def simdensity(eval, E_min, E_max):
    e = (eval-E_min) / (E_max - E_min)
    return e


def gen_psis_and_eigvs(N, H, num_psis):
    """
    Finds the initial state psi for a given Hamiltonian.
    "basis" is a function.
    """
    global index
    evals, evecs = np.linalg.eigh(H.toarray())
    E_max = evals[-1]
    E_min = evals[0]
    # E_max = eigsh(H, k=1, which='LA', maxiter=1e6, return_eigenvectors=False)
    # E_min = eigsh(H, k=1, which='SA', maxiter=1e6, return_eigenvectors=False)
    # E = np.append(E_min, E_max)
    print("min", E_min, "max", E_max)
    evecs = lil_matrix(evecs, dtype=complex)
    # evals.sort()
    zero_Sz_basis_count = int(round(comb(N, 0.5 * N)))
    error = False
    counter = 0
    num_good = 0
    psilist = []
    eigenvalues = []

    for i in range(evecs.get_shape()[0]):
        e = simdensity(evals[i], E_min, E_max)
        if abs(e - 0.5) < .03:
            psi = recast(N, evecs[i])
            psilist.append(psi)
            eigenvalues.append(evals[i])
            num_good += 1
        counter += 1
        if num_good >= num_psis:
            print("Found Enough: ", num_good)
            break
        if counter >= zero_Sz_basis_count:
            print("Not Enough Found")
            error = True
            break
    # # Loop to find a list of suitable eigenstates
    # for psi in evecs:
    #     # Make sure psi's energy density is very close to 0.5.
    #     e, ev = energy_density(psi, H, E)
    #     if abs(e - 0.5) < .01:
    #         psi = recast(N, psi)
    #         psilist.append(psi)
    #         num_good += 1
    #
    #     counter += 1
    #     # Break if enough states are found.
    #     if num_good >= num_psis:
    #         print("Found Enough: ", num_good)
    #         break
    #     # Display an error message when no suitable state is found.
    #     if counter >= zero_Sz_basis_count:
    #         print("Not Enough Found")
    #         error = True
    #         break
    #
    # # Select num_psis amount of eigenvalues near zero energy
    # for i in range(len(evals)):
    #     if abs(((evals[i] - E[0]) / (E[1] - E[0])) - .5) < .01:
    #         for k in range(num_psis):
    #             eigenvalues.append(evals[i+k])
    #         break
    #     if i >= len(evals):
    #         print("problemo")
    #         break
    eigenvalues.sort()
    return H, psilist, eigenvalues, error
