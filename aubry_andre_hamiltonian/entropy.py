import quantum_module as qm
from aubry_andre_common import get_state_blk_full, spin2z, \
    average_adj_gap_ratio, average_vn_entropy, recast, get_state_blk
import numpy as np
from scipy.sparse.linalg import expm_multiply, expm
import aubry_andre_block_H as aubryH


def plot_entropy_time_evo_log(spin, N, h, c, phi, time_range_lower_lim,
                              time_range_upper_lim, sample_size):
    D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    init_delta_t, r = qm.get_init_delta_t(time_range_lower_lim,
                                          time_range_upper_lim, sample_size)
    # The spin 0 block of H
    H = aubryH.blk_full(N, h, c, 0, phi).tocsc()
    psi, error = get_state_blk(H, N)

    if not error:
        # Plot the first point which does not require time evolution.
        psi_long = recast(N, psi)           # psi in the full spin basis
        psi_tz = spin2z(D, N, psi_long)     # psi in the total Sz basis
        entropy_plot[0] += qm.get_vn_entropy(psi_tz, spin, N, mode='eqsplit')

        # Plot the second point which requires the first time evolution.
        current_delta_t = init_delta_t
        psi_tevo_short = expm_multiply(-1j * H * current_delta_t, psi)
        psi_tevo_long = recast(N, psi_tevo_short)
        psi_tevo_tz = spin2z(D, N, psi_tevo_long)
        entropy_plot[1] += qm.get_vn_entropy(psi_tevo_tz,
                                             spin, N, mode='eqsplit')

        # Plot the rest of the points with time evolution.
        for plot_point in range(2, sample_size):
            delta_delta_t = qm.get_delta_delta_t(time_range_lower_lim,
                                                 plot_point, r)
            current_delta_t += delta_delta_t
            psi_tevo_short = expm_multiply(-1j * H * current_delta_t,
                                           psi_tevo_short)
            psi_tevo_long = recast(N, psi_tevo_short)
            psi_tevo_tz = spin2z(D, N, psi_tevo_long)
            entropy_plot[plot_point] += qm.get_vn_entropy(psi_tevo_tz,
                                                          spin, N,
                                                          mode='eqsplit')
    return entropy_plot, error


def plot_entropy_time_evo_lin(spin, N, h, c, phi, time_range_lower_lim,
                              time_range_upper_lim, sample_size):
    D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    delta_t = (time_range_upper_lim - time_range_lower_lim) / (sample_size - 1)

    # The spin 0 block of H
    H = aubryH.blk_full(N, h, c, 0, phi).tocsc()
    psi, error = get_state_blk(H, N)

    if not error:
        # Plot the first point which does not require time evolution.
        psi_long = recast(N, psi)            # psi in the full spin basis
        psi_tz = spin2z(D, N, psi_long)      # psi in the total Sz basis
        entropy_plot[0] += qm.get_vn_entropy(psi_tz, spin, N, mode='eqsplit')

        U = expm(-1j * H * delta_t)
        psi_time_evolved = psi
        # Plot the rest of the points.
        for plot_point in range(1, sample_size):
            psi_time_evolved = U * psi_time_evolved
            # Rewrite the time evolved state in the total Sz basis
            #  before passing it onto the entropy function.
            psi_tevo_long = recast(N, psi_time_evolved)
            psi_time_evolved_tz = spin2z(D, N, psi_tevo_long)
            entropy_plot[plot_point] = qm.get_vn_entropy(psi_time_evolved_tz,
                                                         spin, N,
                                                         mode='eqsplit')
    return entropy_plot, error


def plot_entropy_and_gap_var_h(spin, N, hmin, hmax, c, phi, sample_size):
    # This is just a rough outline... FIX THIS LATER
    h_list = np.linspace(hmin, hmax, sample_size)
    entropy_plot = np.zeros(sample_size)
    adj_gap_ratio_plot = np.zeros(sample_size)

    for h in h_list:
        psi_list, eigenvalues, error = get_state_blk_full(h, N)
        entropy_plot[h] = average_vn_entropy(psi_list, spin, N)
        adj_gap_ratio_plot[h] = average_adj_gap_ratio(eigenvalues)
    return entropy_plot, adj_gap_ratio_plot, error
