import quantum_module as qm
from aubry_andre_common import get_state
import numpy as np
from scipy.sparse.linalg import expm_multiply, expm


def plot_entropy_time_evo_log(spin, N, h, c, phi, time_range_lower_lim,
                              time_range_upper_lim, sample_size):
    D = int(2*spin+1)**N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    init_delta_t, r = qm.get_init_delta_t(time_range_lower_lim,
                                          time_range_upper_lim, sample_size)
    H, psi, exit_status = get_state(Sx, Sy, Sz, N, h, c, phi)

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
    D = int(2*spin+1)**N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    delta_t = (time_range_upper_lim-time_range_lower_lim)/(sample_size-1)
    H, psi, exit_status = get_state(Sx, Sy, Sz, N, h, c, phi)

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
    # This is just a rough outline... FIX THIS LATER
    h_list = np.linspace(h_lower_lim, h_upper_lim, sample_size)
    D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    adj_gap_ratio_plot = np.zeros(sample_size)

    for h in h_list:
        eigenvalues, psilist, exit_status = get_0_states(Sx, Sy, Sz, N, h, c, phi)
        for psi in psilist:
            entropy_plot[h] += qm.get_vn_entropy(psi, spin, N, mode='eqsplit')
        entropy_plot[h] /= np.shape(psilist)[1]
        delta_n_list = np.diff(eigenvalues)
        for x in xrange(0, len(delta_n_list) - 2):
            adj_gap_ratio_plot[h] += min(delta_n_list[x],delta_n_list[x + 1]) / max(delta_n_list[x],delta_n_list[x + 1])
        adj_gap_ratio_plot[h] /= np.shape(delta_n_list)[0] - 1
    return entropy_plot, adj_gap_ratio_plot, exit_status