import quantum_module as qm
import aubry_andre_common as aubryC
import numpy as np
from scipy.sparse.linalg import expm_multiply, expm
import aubry_andre_block_H as aubryH


def plot_entropy_time_evo_log(spin, N, h, c, phi, time_range_lower_lim,
                              time_range_upper_lim, sample_size):
    """
    This function plots the time evolution of von Neuman entropy over a
    logarithmic time axis.

    Args: "spin" is the spin of the individual particles
          "N" is the system size
          "h" is the strength of the pseudo-random field
          "c" is the angular frequency of the field
          "phi" is the phase shift
          "time_range_lower_lim" is the first point in the plot, in time
          "time_range_upper_lim" is the last point in the plot, in time
          "sample_size" is the number points to plot
    Returns: "imbalance_plot" is a list of values to be plotted.
             "error" is the status of the state choosing function that
             is called from this function. If "error" is True, then no
             state of a zero total <Sz> with an energy density could be found
             for the current configuration.
    """
    D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    init_delta_t, r = qm.get_init_delta_t(time_range_lower_lim,
                                          time_range_upper_lim, sample_size)
    # The spin 0 block of H
    H = aubryH.blk_full(N, h, c, 0, phi).tocsc()
    psi, error = aubryC.get_state_blk(H, N)

    if not error:
        # Plot the first point which requires a different kind of time
        #  evolution.
        psi_tevo_short = expm_multiply(-1j * H * time_range_lower_lim, psi)
        psi_long = aubryC.recast(N, psi_tevo_short)
        psi_tz = aubryC.spin2z(D, N, psi_long)   # psi in the total Sz basis
        entropy_plot[0] += qm.get_vn_entropy(psi_tz, spin, N, mode='eqsplit')

        # Plot the rest of the points with time evolution.
        for plot_point in range(1, sample_size):
            if plot_point == 1:
                current_delta_t, r = qm.get_init_delta_t(time_range_lower_lim,
                                                         time_range_upper_lim,
                                                         sample_size)
            elif plot_point > 1:
                delta_delta_t = qm.get_delta_delta_t(time_range_lower_lim,
                                                     plot_point, r)
                current_delta_t += delta_delta_t

            psi_tevo_short = expm_multiply(-1j * H * current_delta_t,
                                           psi_tevo_short)
            psi_tevo_long = aubryC.recast(N, psi_tevo_short)
            psi_tevo_tz = aubryC.spin2z(D, N, psi_tevo_long)
            entropy_plot[plot_point] += qm.get_vn_entropy(psi_tevo_tz,
                                                          spin, N,
                                                          mode='eqsplit')
    return entropy_plot, error


def plot_entropy_time_evo_lin(spin, N, h, c, phi, time_range_lower_lim,
                              time_range_upper_lim, sample_size):
    """
    This function plots the time evolution of von Neuman entropy over a
    linear time axis.

    Args: "spin" is the spin of the individual particles
          "N" is the system size
          "h" is the strength of the pseudo-random field
          "c" is the angular frequency of the field
          "phi" is the phase shift
          "time_range_lower_lim" is the first point in the plot, in time
          "time_range_upper_lim" is the last point in the plot, in time
          "sample_size" is the number points to plot
    Returns: "imbalance_plot" is a list of values to be plotted.
             "error" is the status of the state choosing function that
             is called from this function. If "error" is True, then no
             state of a zero total <Sz> with an energy density could be found
             for the current configuration.
    """
    D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(sample_size)
    delta_t = (time_range_upper_lim - time_range_lower_lim) / (sample_size - 1)

    # The spin 0 block of H
    H = aubryH.blk_full(N, h, c, 0, phi).tocsc()
    psi, error = aubryC.get_state_blk(H, N)

    if not error:
        # Plot the first point which requires a special kind of time evolution.
        psi = expm_multiply(-1j * H * time_range_lower_lim, psi)
        # psi in the full spin basis
        psi_long = aubryC.recast(N, psi)
        psi_tz = aubryC.spin2z(D, N, psi_long)      # psi in the total Sz basis
        entropy_plot[0] += qm.get_vn_entropy(psi_tz, spin, N, mode='eqsplit')

        U = expm(-1j * H * delta_t)
        psi_time_evolved = psi
        # Plot the rest of the points.
        for plot_point in range(1, sample_size):
            psi_time_evolved = U * psi_time_evolved
            # Rewrite the time evolved state in the total Sz basis
            #  before passing it onto the entropy function.
            psi_tevo_long = aubryC.recast(N, psi_time_evolved)
            psi_time_evolved_tz = aubryC.spin2z(D, N, psi_tevo_long)
            entropy_plot[plot_point] = qm.get_vn_entropy(psi_time_evolved_tz,
                                                         spin, N,
                                                         mode='eqsplit')
    return entropy_plot, error


def entropy_agr_vs_h(spin, N, hmin, hmax, c, phi, sample_size, num_psis):
    # This is just a rough outline... FIX THIS LATER
    h_list = np.linspace(hmin, hmax, sample_size)
    entropy_plot = np.zeros(sample_size)
    adj_gap_ratio_plot = np.zeros(sample_size)
    error = False
    for i in range(len(h_list)):
        print("h", i + 1, ": ", h_list[i])
        H = aubryH.blk_full(N, h_list[i], c, 0, phi).tocsc()
        H, psis, eigvs= aubryC.gen_eigenpairs(N, H, num_psis)
        entropy_plot[i] = aubryC.average_vn_entropy(psis, spin, N)
        adj_gap_ratio_plot[i] = aubryC.average_adj_gap_ratio(eigvs)
    return entropy_plot, adj_gap_ratio_plot, h_list, error


def plot_ent_agr_avg_phi(spin, N, hmin, hmax, hsamples, c, num_psis,
                         phisample):
    """
    to be fixed later, runs plot_entropy_and_gap_var_h over many phi and avgs
    :param spin:
    :param N:
    :param hmin:
    :param hmax:
    :param hsamples:
    :param c:
    :param num_psis:
    :param phisample:
    :return:
    """
    phi_list = np.linspace(0, 2 * np.pi, phisample)
    phi_list = phi_list[0:-1]
    avg_phi_entropy = np.zeros(hsamples)
    avg_phi_agr = np.zeros(hsamples)
    error = False
    for i in range(len(phi_list)):
        print("phi", i + 1, ": ********", phi_list[i], " ********")
        entropy, agr, h_list, error = entropy_agr_vs_h(spin, N, hmin, hmax, c,
                                                       phi_list[i], hsamples,
                                                       num_psis)
        avg_phi_entropy += entropy
        avg_phi_agr += agr
    avg_phi_entropy /= len(phi_list) * N
    avg_phi_agr /= len(phi_list)
    return avg_phi_entropy, avg_phi_agr, h_list, error
