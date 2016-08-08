import quantum_module as qm
from aubry_andre_block_H import aubry_andre_H_off_diag, aubry_andre_H_diag
from aubry_andre_common import get_state_blk
import numpy as np
from scipy import io
from scipy.sparse.linalg import expm_multiply, expm
import os


def plot_entropy_time_evo_log(spin, N, h, c, phi, time_range_lower_lim,
                              time_range_upper_lim, sample_size):
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    init_delta_t, r = qm.get_init_delta_t(time_range_lower_lim,
                                          time_range_upper_lim, sample_size)

    # Check and see if the off diagonal matrix is saved. If so, load from disk.
    if os.path.isfile('block_H_off_diag_' + str(N) + 'spins.mat'):
        H_off_diag = io.loadmat('block_H_off_diag_' +
                                str(N) + 'spins.mat')['i']
    else:
        H_off_diag = aubry_andre_H_off_diag(Sx, Sy, Sz, N)
        H_off_diag += H_off_diag.transpose()
        io.savemat('block_H_off_diag_' + str(N) + 'spins', {'i': H_off_diag})
    H_diag = aubry_andre_H_diag(Sx, Sy, Sz, N, h, c, phi)
    H = H_diag + H_off_diag
    H = H.tolil()

    psi, exit_status = get_state_blk(H, N)
    if not exit_status:
        # Plot the first point which does not require time evolution.
        entropy_plot[0] += qm.get_vn_entropy(psi, spin, N, mode='eqsplit')

        # Plot the second point which requires the first time evolution.
        current_delta_t = init_delta_t
        psi_time_evolved = expm_multiply(-1j*H*current_delta_t, psi)
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
    return entropy_plot, exit_status


def plot_entropy_time_evo_lin(spin, N, h, c, phi, time_range_lower_lim,
                              time_range_upper_lim, sample_size):
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    delta_t = (time_range_upper_lim-time_range_lower_lim)/(sample_size-1)

    # Check and see if the off diagonal matrix is saved. If so, load from disk.
    if os.path.isfile('block_H_off_diag_' + str(N) + 'spins.mat'):
        H_off_diag = io.loadmat('block_H_off_diag_' +
                                str(N) + 'spins.mat')['i']
    else:
        H_off_diag = aubry_andre_H_off_diag(Sx, Sy, Sz, N)
        H_off_diag += H_off_diag.transpose()
        io.savemat('block_H_off_diag_' + str(N) + 'spins', {'i': H_off_diag})
    H_diag = aubry_andre_H_diag(Sx, Sy, Sz, N, h, c, phi)
    H = H_diag + H_off_diag

    psi, exit_status = get_state_blk(H, N)
    if not exit_status:
        U = expm(-1j*H*delta_t)

        # Plot the first point which does not require time evolution.
        entropy_plot[0] += qm.get_vn_entropy(psi, spin, N, mode='eqsplit')

        psi_time_evolved = psi
        # Plot the rest of the points.
        for plot_point in range(1, sample_size):
            psi_time_evolved = U*psi_time_evolved
            entropy_plot[plot_point] = qm.get_vn_entropy(psi_time_evolved,
                                                         spin, N,
                                                         mode='eqsplit')
    return entropy_plot, exit_status


def plot_entropy_var_h(spin, N, h_lower_lim, h_upper_lim, c, phi, sample_size):
    h_list = np.linspace(h_lower_lim, h_upper_lim, sample_size)
