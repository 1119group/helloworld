import quantum_module as qm
import aubry_andre_common as aubryC
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import expm_multiply, expm
import aubry_andre_block_H as aubryH
import time
from timer import Timer


def entropy_time_plot(N, H, delta_ts, start_time=0):
    """
    This function plots the time evolution of von Neuman entropy
    using exact diagonalization.

    Args: "spin" is the spin of the individual particles
          "N" is the system size
          "h" is the strength of the pseudo-random field
          "c" is the angular frequency of the field
          "phi" is the phase shift
          "delta_ts" is a list/array of delta t to be passed on the function
    Returns: "imbalance_plot" is a list of values to be plotted.
             "error" is the status of the state choosing function that
             is called from this function. If "error" is True, then no
             state of a zero total <Sz> with an energy density could be found
             for the current configuration.
    """
    spin = 0.5
    # D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    points = len(delta_ts) + 1
    entropy_plot = np.zeros(points)

    # Use exact diagonalization for small systems.
    psi, error = aubryC.get_state_blk(H, N)
    H = H.toarray()
    E, V = np.linalg.eigh(H)
    tm = aubryC.TimeMachine(E, V, psi)

    if not error:
        # Plot the rest of the points with time evolution.
        for plot_point in range(points):
            if plot_point == 0:
                psi_t = tm.evolve(start_time)
            else:
                psi_t = tm.evolve(delta_ts[plot_point - 1])
            # psi_long = lil_matrix(aubryC.recast(N, psi_t))
            psi_tz = aubryC.spin2z_blk(N, psi_t)
            entropy_plot[plot_point] += qm.get_vn_entropy(psi_tz, spin, N,
                                                          mode='eqsplit')
    return entropy_plot, error


def ent_agr_var_plots(spin, N, hmin, hmax, points, c, num_psis, phis):
    """
    This function creates data dumps for plotting of eigenstate entropy,
    adjusted gap ratio, and variance. Also data for histograms
    :param spin: is the spin of the individual particles
    :param N: is the system size
    :param hmin: minimum h to plot
    :param hmax: maximum h to plot
    :param points: number of h to plot
    :param c: quasirandom factor
    :param num_psis: number of eigenstates to find near a specified energy
    :param phis: number of phases to average over
    :return: dumped data files
    """
    h_list = np.linspace(hmin, hmax, points)
    phi_list = np.linspace(0, 0.5 * np.pi, phis + 1)
    phi_list = phi_list[0:-1]
    Sz_tot = aubryC.half_chain_Sz(N)
    Sz_tot2 = Sz_tot**2
    ent_plt = np.zeros(points)
    var_plt = np.zeros(points)
    agr_plt = np.zeros(points)
    ent_std = np.zeros(points)
    var_std = np.zeros(points)
    agr_std = np.zeros(points)
    timer = Timer(points*phis, mode='average')
    for i in range(points):
        ent_lst_full = np.array([])
        var_lst_full = np.array([])
        agr_lst_full = np.array([])
        # eig_lst_full = ent_lst_full
        for k in range(phis):
            H = aubryH.blk_full(N, h_list[i], c, 0, phi_list[k])
            try:
                H, psis, eigvs = aubryC.gen_eigenpairs(N, H, num_psis)
            except:
                print("Configuration failed to converge: H ", h_list[i], " Phi ", phi_list[k])
            ent_lst, var_lst = \
                aubryC.ent_var_lst(psis, spin, N, Sz_tot, Sz_tot2)
            eigvs = np.sort(eigvs)
            agr_lst = aubryC.adj_gap_ratio(eigvs)
            ent_lst_full = np.append(ent_lst_full, ent_lst)
            var_lst_full = np.append(var_lst_full, var_lst)
            agr_lst_full = np.append(agr_lst_full, agr_lst)
            # eig_lst_full = np.append(eig_lst_full, eigvs)
            timer.progress()
        ent_plt[i] = np.mean(ent_lst_full)
        ent_std[i] = np.std(ent_lst_full)
        var_plt[i] = np.mean(var_lst_full)
        var_std[i] = np.std(var_lst_full)
        agr_plt[i] = np.mean(agr_lst_full)
        agr_std[i] = np.std(agr_lst_full)
        ent_hst_file = 'DATA/ent_hst_L' + str(N) + '_h' + str(
            h_list[i]) + '_c' + str(round(c, 2)) + '.txt'
        var_hst_file = 'DATA/var_hst_L' + str(N) + '_h' + str(
            h_list[i]) + '_c' + str(round(c, 2)) + '.txt'
        agr_hst_file = 'DATA/agr_hst_L' + str(N) + '_h' + str(
            h_list[i]) + '_c' + str(round(c, 2)) + '.txt'
        # eig_hst_file = 'eig_hst_L' + str(N) + '_h' + str(
        #     h_list[i]) + '_c' + str(round(c, 2)) + '.txt'
        np.savetxt(ent_hst_file, np.transpose(ent_lst_full))
        np.savetxt(var_hst_file, np.transpose(var_lst_full))
        np.savetxt(agr_hst_file, np.transpose(agr_lst_full))
        # np.savetxt(eig_hst_file, eig_lst_full)
    ent_plt /= N
    ent_agr_var_plot_file = 'DATA/plot_data_L' + str(N) + '_c' + str(round(c, 2))\
                            + '_phi' + str(phis) + '.txt'
    np.savetxt(ent_agr_var_plot_file, (h_list, ent_plt, ent_std, agr_plt, agr_std, var_plt, var_std))


###############################################################################
###############################################################################
# Deprecated Code. Leave until testing of the new entropy function is         #
#  complete.                                                                  #
###############################################################################
###############################################################################
def plot_entropy_time_evo_lin(spin, N, h, c, phi, start_time,
                              end_start, points):
    """
    This function plots the time evolution of von Neuman entropy over a
    linear time axis.

    Args: "spin" is the spin of the individual particles
    "N" is the system size
    "h" is the strength of the pseudo-random field
    "c" is the angular frequency of the field
    "phi" is the phase shift
    "start_time" is the first point in the plot, in time
    "end_start" is the last point in the plot, in time
    "points" is the number points to plot
    Returns: "imbalance_plot" is a list of values to be plotted.
    "error" is the status of the state choosing function that
    is called from this function. If "error" is True, then no
    state of a zero total <Sz> with an energy density could be found
    for the current configuration.
    """
    D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(points)
    delta_t = (end_start - start_time) / (points - 1)

    # The spin 0 block of H
    H = aubryH.blk_full(N, h, c, 0, phi).tocsc()
    E, V = np.linalg.eigh(H.toarray())
    psi, error = aubryC.get_state_blk(H, N)
    # psi = psi.toarray()

    if not error:
        # Plot the first point which requires a special kind of time evolution.
        psi = expm_multiply(-1j * H * start_time, psi)
        # psi = aubryC.time_evo_exact_diag(E, V, psi, start_time)
        # psi = lil_matrix(psi)
        # psi in the full spin basis
        psi_long = aubryC.recast(N, psi)
        psi_tz = aubryC.spin2z(D, N, psi_long)      # psi in the total Sz basis
        entropy_plot[0] += qm.get_vn_entropy(psi_tz, spin, N, mode='eqsplit')

        U = expm(-1j * H * delta_t)
        psi_time_evolved = psi
        # Plot the rest of the points.
        for plot_point in range(1, points):
            psi_time_evolved = U * psi_time_evolved
            # psi_time_evolved = aubryC.time_evo_exact_diag(E, V, psi_time_evolved ,delta_t)
            # psi_time_evolved = lil_matrix(psi_time_evolved)
            # Rewrite the time evolved state in the total Sz basis
            #  before passing it onto the entropy function.
            psi_tevo_long = aubryC.recast(N, psi_time_evolved)
            psi_time_evolved_tz = aubryC.spin2z(D, N, psi_tevo_long)
            entropy_plot[plot_point] = qm.get_vn_entropy(psi_time_evolved_tz,
                                                         spin, N,
                                                         mode='eqsplit')
    return entropy_plot, error


def plot_entropy_time_evo_log(spin, N, h, c, phi, start_time,
                              end_start, points):
    """
    This function plots the time evolution of von Neuman entropy over a
    logarithmic time axis.

    Args: "spin" is the spin of the individual particles
          "N" is the system size
          "h" is the strength of the pseudo-random field
          "c" is the angular frequency of the field
          "phi" is the phase shift
          "start_time" is the first point in the plot, in time
          "end_start" is the last point in the plot, in time
          "points" is the number points to plot
    Returns: "imbalance_plot" is a list of values to be plotted.
             "error" is the status of the state choosing function that
             is called from this function. If "error" is True, then no
             state of a zero total <Sz> with an energy density could be found
             for the current configuration.
    """
    D = int(2 * spin + 1) ** N
    Sx, Sy, Sz = qm.init(spin)
    entropy_plot = np.zeros(points)
    init_delta_t, r = qm.get_init_delta_t(start_time,
                                          end_start, points)
    # The spin 0 block of H
    H = aubryH.blk_full(N, h, c, 0, phi).tocsc()
    # Use exact diagonalization for small systems.
    psi, error = aubryC.get_state_blk(H, N)
    dense = False
    if H.get_shape()[0] <= 16:
        dense = True
        H = H.toarray()
        E, V = np.linalg.eigh(H)
        tm = aubryC.time_machine(E, V, psi)

    if not error:
        # Plot the first point which requires a different kind of time
        #  evolution.

        if H.get_shape()[0] <= 16:
            psi_tevo_short = tm.evolve(start_time)
        else:
            psi_tevo_short = expm_multiply(-1j * H * start_time, psi)
        psi_long = aubryC.recast(N, psi_tevo_short)
        psi_tz = aubryC.spin2z(D, N, psi_long)   # psi in the total Sz basis
        entropy_plot[0] += qm.get_vn_entropy(psi_tz, spin, N, mode='eqsplit')

        # Plot the rest of the points with time evolution.
        for plot_point in range(1, points):
            if plot_point == 1:
                current_delta_t, r = qm.get_init_delta_t(start_time,
                                                         end_start,
                                                         points)
            elif plot_point > 1:
                delta_delta_t = qm.get_delta_delta_t(start_time,
                                                     plot_point, r)
                current_delta_t += delta_delta_t

            if dense:
                psi_tevo_short = tm.evolve(current_delta_t)
            else:
                psi_tevo_short = expm_multiply(-1j * H * current_delta_t,
                                               psi_tevo_short)
            psi_tevo_long = aubryC.recast(N, psi_tevo_short)
            psi_tevo_tz = aubryC.spin2z(D, N, psi_tevo_long)
            entropy_plot[plot_point] += qm.get_vn_entropy(psi_tevo_tz,
                                                          spin, N,
                                                          mode='eqsplit')
    return entropy_plot, error


def entropy_agr_vs_h(spin, N, hmin, hmax, c, phi, points, num_psis):
    # This is just a rough outline... FIX THIS LATER
    h_list = np.linspace(hmin, hmax, points)
    entropy_plot = np.zeros(points)
    adj_gap_ratio_plot = np.zeros(points)
    for i in range(len(h_list)):
        H = aubryH.blk_full(N, h_list[i], c, 0, phi).tocsc()
        H, psis, eigvs = aubryC.gen_eigenpairs(N, H, num_psis)
        entropy_plot[i] = aubryC.average_vn_entropy(psis, spin, N)
        adj_gap_ratio_plot[i] = aubryC.average_adj_gap_ratio(eigvs)
    return entropy_plot, adj_gap_ratio_plot, h_list


def entropy_var_vs_h(spin, N, hmin, hmax, c, phi, points, num_psis, Sz_tot, Sz_tot2):
    # replacement function to add variance and keep individual values
    h_list = np.linspace(hmin, hmax, points)
    entropy_plot = np.zeros(points)
    variance_plot = np.zeros(points)
    adj_gap_ratio_plot = np.zeros(points)
    entropy_list_over_h = np.array([])
    variance_list_over_h = np.array([])
    eigvs_list_over_h = np.array([])
    for i in range(len(h_list)):
        H = aubryH.blk_full(N, h_list[i], c, 0, phi).tocsc()
        H, psis, eigvs = aubryC.gen_eigenpairs(N, H, num_psis)
        entropy_plot[i], entropy_list, variance_plot[i], variance_list = \
            aubryC.ent_var_lst(psis, spin, N, Sz_tot, Sz_tot2)
        adj_gap_ratio_plot[i] = aubryC.average_adj_gap_ratio(eigvs)
        entropy_list_over_h = np.append(entropy_list_over_h, entropy_list)
        variance_list_over_h = np.append(variance_list_over_h, variance_list)
        eigvs_list_over_h = np.append(eigvs_list_over_h, eigvs)
    return entropy_plot, entropy_list_over_h, variance_plot, \
           variance_list_over_h, adj_gap_ratio_plot, eigvs_list_over_h, h_list


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
    Sz_tot = aubryC.half_chain_Sz(N)
    Sz_tot2 = Sz_tot**2
    phi_list = np.linspace(0, 2 * np.pi, phisample + 1)
    phi_list = phi_list[0:-1]
    avg_phi_entropy = np.zeros(hsamples)
    avg_phi_agr = np.zeros(hsamples)
    avg_phi_var = np.zeros(hsamples)
    entropy_list_over_phi = np.array([])
    variance_list_over_phi = np.array([])
    eigvs_list_over_phi = np.array([])
    for i in range(len(phi_list)):
        start = time.time()
        print("phi", i + 1, "N", N)
        entropy, entropy_list, variance, variance_list, agr, eigvs, h_list = \
            entropy_var_vs_h(spin, N, hmin, hmax, c, phi_list[i], hsamples,
                             num_psis, Sz_tot, Sz_tot2)
        avg_phi_entropy += entropy
        avg_phi_agr += agr
        avg_phi_var += variance
        entropy_list_over_phi = np.append(entropy_list_over_phi, entropy_list)
        variance_list_over_phi = np.append(variance_list_over_phi, variance_list)
        eigvs_list_over_phi = np.append(eigvs_list_over_phi, eigvs)
        end = time.time()
        elap = end - start
        print("Iteration Time:", elap, "exp total:", elap * len(phi_list))
    avg_phi_var /= len(phi_list)
    avg_phi_entropy /= len(phi_list)
    avg_phi_agr /= len(phi_list)
    return avg_phi_entropy, entropy_list_over_phi, avg_phi_agr, variance, \
           variance_list_over_phi, eigvs_list_over_phi, h_list
