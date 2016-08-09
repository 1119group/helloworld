import quantum_module as qm
from aubry_andre_H import aubry_andre_H
import aubry_andre_block_H as aubryH
import numpy as np
from scipy.sparse import dok_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.misc import comb
from scipy import io
import os


def spin2z(D, N, psi):
    """
    Rewrites a given state psi from the spin basis (the basis of
    the block diagonalized Hamiltonian) to the conventional
    Sz product basis.
    Only works for zero total <Sz>.
    """
    psi_tz = lil_matrix((D, 1), dtype=complex)
    j_max = int(round(0.5 * N))
    blk_sz = int(round(comb(N, j_max)))
    blk_pos = int(round(0.5 * (D - blk_sz)))
    basis_set_0, basis_dict_0 = aubryH.basis_set(N, blk_sz, j_max, 0)
    for i in basis_dict_0:
        psi_tz[i, 0] = psi[basis_dict_0[i] + blk_pos, 0]

    psi_tz = psi_tz.tocsc()
    return psi_tz


def spin2z_0(D, N, psi):
    """
    Rewrites a given state psi from the spin basis (the basis of
    the block diagonalized Hamiltonian) to the conventional
    Sz product basis.
    This version takes in a truncated psi in the zero spin basis and
    reconstruct the full psi vector in the Sz product basis.
    Only works for zero total <Sz>.
    """
    global basis_set_0, basis_dict_0
    psi_tz = lil_matrix((D, 1), dtype=complex)
    j_max = int(round(0.5 * N))
    blk_sz = int(round(comb(N, j_max)))
    basis_set_0, basis_dict_0 = aubryH.basis_set(N, blk_sz, j_max, 0)
    for i in basis_dict_0:
        psi_tz[i, 0] = psi[basis_dict_0[i], 0]

    psi_tz = psi_tz.tocsc()
    return psi_tz


def generate_H(Sx, Sy, Sz, N, h, c, phi):
    """
    A function to put together the pieces of a Hamiltonian and
    generate a full Hamiltonian.
    """
    # Check and see if the off diagonal matrix is saved.
    #  If so, load from disk.
    if os.path.isfile('block_H_off_diag_' + str(N) + 'spins.mat'):
        H_off_diag = io.loadmat('block_H_off_diag_' +
                                str(N) + 'spins.mat')['i']
    else:
        H_off_diag = aubryH.aubry_andre_H_off_diag(Sx, Sy, Sz, N)
        H_off_diag += H_off_diag.transpose()
        io.savemat('block_H_off_diag_' + str(N) + 'spins', {'i': H_off_diag})
    H_diag = aubryH.aubry_andre_H_diag(Sx, Sy, Sz, N, h, c, phi)
    H = H_diag + H_off_diag
    return H


def energy_density(psi, H, E):
    exp_val = psi.conjtransp().dot(H.dot(psi))
    e = (exp_val[0, 0] - E[0]) / (E[-1] - E[0])
    return e


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
    """Returns a state psi in the spin basis."""
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
        e = energy_density(psi_0, H, E)
        if abs(e - 0.5) < 0.001:
            break

        counter += 1
        # Display an error message when no suitable state is found.
        if counter > zero_Sz_basis_count:
            error = True
            break
    return H, psi_0, error


def get_state(Sx, Sy, Sz, N, h, c, phi=0, J=1):
    H = aubry_andre_H(Sx, Sy, Sz, N, h, c, phi)
    H, psi_0, error = init_state(N, H, z_basis)
    return H, psi_0, error


def get_0_state_blk(H, N):
    """
    "H" is only the center block of the Hamiltonian.
    """
    H, psi, error = init_state(N, H, spin_basis)
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
        adj_gap_ratio += min(delta_n_list[x],delta_n_list[x + 1]) /\
                         max(delta_n_list[x],delta_n_list[x + 1])
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
    avg_vn_entropy /= np.shape(list_of_states)[1]
    return avg_vn_entropy
