import quantum_module as qm
import aubry_andre_block_H as aubryH
from aubry_andre_common import get_0_state_blk
import numpy as np
from scipy import io
from scipy.misc import comb
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import expm_multiply, expm
import os


def spin2z(D, N, psi):
    """
    Rewrites a given state psi from the spin basis (the basis of
    the block diagonalized Hamiltonian) to the conventional
    Sz product basis.
    """
    psi_tz = lil_matrix((D, 1), dtype=complex)
    j_max = int(round(0.5 * N))
    blk_sz = int(round(comb(N, j_max)))
    basis_set_0, basis_dict_0 = aubryH.basis_set(N, blk_sz, j_max, 0)
    for i in basis_dict_0:
        psi_tz[i] = psi[basis_dict_0[i], 0]

    psi_tz = psi_tz.tocsc()
    return psi_tz


def generate_H(Sx, Sy, Sz, N, h, c, phi):
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


def plot_entropy_time_evo_log(spin, N, h, c, phi, time_range_lower_lim,
                              time_range_upper_lim, sample_size):
    D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    init_delta_t, r = qm.get_init_delta_t(time_range_lower_lim,
                                          time_range_upper_lim, sample_size)
    H = generate_H(Sx, Sy, Sz, N, h, c, phi)
    psi, error = get_0_state_blk(H, N)

    # Rewrite psi in the Sz basis
    psi_tz = spin2z(D, N, psi)

    if not error:
        # Plot the first point which does not require time evolution.
        entropy_plot[0] += qm.get_vn_entropy(psi_tz, spin, N, mode='eqsplit')

        # Plot the second point which requires the first time evolution.
        current_delta_t = init_delta_t
        psi_time_evolved = expm_multiply(-1j*H*current_delta_t, psi_tz)
        entropy_plot[1] += qm.get_vn_entropy(psi_time_evolved,
                                             spin, N, mode='eqsplit')

        # Plot the rest of the points with time evolution.
        for plot_point in range(2, sample_size):
            delta_delta_t = qm.get_delta_delta_t(time_range_lower_lim,
                                                 plot_point, r)
            current_delta_t += delta_delta_t
            psi_time_evolved = expm_multiply(-1j*H*current_delta_t,
                                             psi_time_evolved)
            entropy_plot[plot_point] += qm.get_vn_entropy(psi_time_evolved,
                                                          spin, N,
                                                          mode='eqsplit')
    return entropy_plot, error


def plot_entropy_time_evo_lin(spin, N, h, c, phi, time_range_lower_lim,
                              time_range_upper_lim, sample_size):
    D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    delta_t = (time_range_upper_lim-time_range_lower_lim)/(sample_size-1)
    H = generate_H(Sx, Sy, Sz, N, h, c, phi)
    psi, error = get_0_state_blk(H, N)

    # Rewrite psi in the Sz basis
    psi_tz = spin2z(D, N, psi)

    if not error:
        U = expm(-1j*H*delta_t)

        # Plot the first point which does not require time evolution.
        entropy_plot[0] += qm.get_vn_entropy(psi_tz, spin, N, mode='eqsplit')

        psi_time_evolved = psi_tz
        # Plot the rest of the points.
        for plot_point in range(1, sample_size):
            psi_time_evolved = U*psi_time_evolved
            entropy_plot[plot_point] = qm.get_vn_entropy(psi_time_evolved,
                                                         spin, N,
                                                         mode='eqsplit')
    return entropy_plot, error


def plot_entropy_var_h(spin, N, h_lower_lim, h_upper_lim, c, phi, sample_size):
    # This is just a rough outline... FIX THIS LATER
    h_list = np.linspace(h_lower_lim, h_upper_lim, sample_size)
    D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    adj_gap_ratio_plot = np.zeros(sample_size)

    for h in h_list:
        psilist, error = get_0_states(Sx, Sy, Sz, N, h, c, phi)
        for psi in psilist:
            entropy_plot[h] += qm.get_vn_entropy(psi, spin, N, mode='eqsplit')
        entropy_plot[h] /= len(psilist)
    return entropy_plot, error
        eigenvalues, psilist, exit_status = get_0_states(Sx, Sy, Sz, N, h, c, phi)
        for psi in psilist:
            entropy_plot[h] += qm.get_vn_entropy(psi, spin, N, mode='eqsplit')
        entropy_plot[h] /= np.shape(psilist)[1]
        delta_n_list = np.diff(eigenvalues)
        for x in xrange(0, len(delta_n_list) - 2):
            adj_gap_ratio_plot[h] += min(delta_n_list[x],delta_n_list[x + 1]) / max(delta_n_list[x],delta_n_list[x + 1])
        adj_gap_ratio_plot[h] /= np.shape(delta_n_list)[0] - 1
    return entropy_plot, adj_gap_ratio_plot, exit_status
