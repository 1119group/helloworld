import quantum_module as qm
import aubry_andre_block_H as aubryH
import aubry_andre_common as aubryC
import numpy as np
from scipy.sparse.linalg import expm


def plot_imbalance_time_evo_log(spin, N, h, c, phi, time_range_lower_lim,
                                time_range_upper_lim, sample_size):
    """
    This function plots the time evolution of spin imbalance
    Σ<psi|Sz(0)Sz(t)|psi> over a logarithmic time axis.

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
    Sx, Sy, Sz = qm.init(spin)
    D = 2**N
    H = aubryH.blk_full(N, h, c, 0, phi).tocsc()
    psi, error = aubryC.get_state_blk(H, N)

    # So in a nutshell, the following process is all carried out in the
    #  block spin basis. The Pauli Sz spin operator is rearranged into
    #  the block spin basis and only the center block (spin 0 block) is
    #  sliced out and kept. Everything else happens as usual.
    if not error:
        imbalance_plot = np.zeros(sample_size)
        psi_Szs = np.empty(N, dtype=object)
        full_Szs = np.empty(N, dtype=object)
        ctr_blk_sz = H.get_shape()[0]
        shift = int(round(0.5 * (D - ctr_blk_sz)))

        # Plot the first point which requires a different kind of time
        #  evolution and set up the rest of the calculations by storing
        #  all the Sz operators and the <psi|Sz(0) row vectors.
        U_delta_t_0 = expm(-1j * H * time_range_lower_lim)
        U_delta_t_dag_0 = U_delta_t_0.transpose().conjugate()
        psi_tevo = U_delta_t_0.dot(psi)
        for k in range(N):
            Sz_full_k = qm.get_full_matrix(Sz, k, N)
            # Rewrite Sz_full_k into the block Hamiltonian spin basis.
            Sz_full_k_sb = aubryC.Sz2spin_basis(N, Sz_full_k)
            # Slice out the center block which is the only part that matters.
            Sz_full_k_sb = Sz_full_k_sb[shift:shift + ctr_blk_sz,
                                        shift:shift + ctr_blk_sz]
            psi_Sz = psi.transpose().conjugate().dot(Sz_full_k_sb)
            psi_Sz_tevo = psi_Sz.dot(U_delta_t_dag_0)
            full_Szs[k] = Sz_full_k_sb.copy()
            psi_Szs[k] = psi_Sz_tevo.copy()
            imbalance_plot[0] += np.real(psi_Sz_tevo.dot(
                                         Sz_full_k_sb.dot(psi_tevo))[0, 0])

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

            U_delta_t = expm(-1j * H * current_delta_t)
            U_delta_t_dag = U_delta_t.transpose().conjugate()
            psi_tevo = U_delta_t.dot(psi_tevo)
            for k in range(N):
                psi_Szs[k] = psi_Szs[k].dot(U_delta_t_dag)
                imbalance_plot[plot_point] += np.real(psi_Szs[k].dot(
                    full_Szs[k].dot(
                        psi_tevo))[0, 0])
    return 4 / N * imbalance_plot, error


def plot_imbalance_time_evo_lin(spin, N, h, c, phi, time_range_lower_lim,
                                time_range_upper_lim, sample_size):
    """
    This function plots the time evolution of spin imbalance
    Σ<psi|Sz(0)Sz(t)|psi> over a linear time axis.

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
    D = 2**N
    Sx, Sy, Sz = qm.init(spin)
    imbalance_plot = np.zeros(sample_size)
    delta_t = (time_range_upper_lim - time_range_lower_lim) / (sample_size - 1)

    # The spin 0 block of H
    H = aubryH.blk_full(N, h, c, 0, phi).tocsc()
    psi, error = aubryC.get_state_blk(H, N)

    # So in a nutshell, the following process is all carried out in the
    #  block spin basis. The Pauli Sz spin operator is rearranged into
    #  the block spin basis and only the center block (spin 0 block) is
    #  sliced out and kept. Everything else happens as usual.
    if not error:
        U_delta_t = expm(-1j * H * delta_t)
        U_delta_t_dag = U_delta_t.transpose().conjugate()
        psi_Szs = np.empty(N, dtype=object)
        # full_Szs will be in the block H spin basis
        full_Szs = np.empty(N, dtype=object)
        ctr_blk_sz = H.get_shape()[0]
        shift = int(round(0.5 * (D - ctr_blk_sz)))

        # Plot the first point which does not require time evolution.
        for k in range(N):
            Sz_full_k = qm.get_full_matrix(Sz, k, N)
            Sz_full_k_sb = aubryC.Sz2spin_basis(N, Sz_full_k)
            # Slice out the center block which is the only part that matters.
            Sz_full_k_sb = Sz_full_k_sb[shift:shift + ctr_blk_sz,
                                        shift:shift + ctr_blk_sz]
            psi_Sz = psi.transpose().conjugate().dot(Sz_full_k_sb)
            psi_Szs[k] = psi_Sz.copy()
            full_Szs[k] = Sz_full_k_sb.copy()
            imbalance_plot[0] += np.real(psi_Sz.dot(
                                         Sz_full_k_sb.dot(psi))[0, 0])

        # Plot the rest of the points.
        for plot_point in range(1, sample_size):
            psi = U_delta_t_dag.dot(psi)
            for k in range(N):
                psi_Szs[k] = psi_Szs[k].dot(U_delta_t)
                imbalance_plot[plot_point] += np.real(psi_Szs[k].dot(
                                                      full_Szs[k].dot(
                                                          psi))[0, 0])
    return 4 / N * imbalance_plot, error
