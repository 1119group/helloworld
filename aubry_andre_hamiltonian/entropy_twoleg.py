import quantum_module as qm
import aubry_andre_common as aubryC
import numpy as np
import aubry_andre_block_twoleg_H as hm


def spin2z_nocompat(N, psi, psi_tz, current_j):
    to_ord = hm.create_complete_basis(N, current_j)[2]
    for i in psi.nonzero()[0]:
        dec = to_ord[i]
        psi_tz[2 ** N - 1 - dec, 0] = psi[i, 0]
    return psi_tz


def spin2z(N, psi, current_j=0):
    return qm.sp_den_col_row_compat(N, psi, spin2z_nocompat, [current_j])


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
            psi_tz = spin2z(N, psi_t)
            entropy_plot[plot_point] += qm.get_vn_entropy(psi_tz, spin, N,
                                                          mode='eqsplit')
    return entropy_plot, error
