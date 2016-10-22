import quantum_module as qm
import scipy.sparse as sp
import aubry_andre_common as common
import numpy as np


def reorder_basis(N, psi_diag, to_ord, current_j=0):
    """
    Reorders the basis of a vector from one arranged by their total <Sz>
    to one that results from tensor products.

    Args: "N" System size
          "psi_diag" State in a block diagonalized basis arrangement
          "to_ord" A Dictionary the maps the total <Sz> arrangement to tensor
          product arrangement
          "current_j" Total <Sz>
    Returns: Sparse Nx1 matrix (vector)
    """
    psi_ord = sp.lil_matrix((2 ** N, 1), dtype=complex)
    for i in psi_diag.nonzero()[0]:
        dec = to_ord[i]
        psi_ord[2 ** N - 1 - dec, 0] = psi_diag[i, 0]
    return psi_ord


def entropy_time_plot(N, H, to_ord, delta_ts, start_time=0):
    """
    This function plots the time evolution of von Neuman entropy
    using exact diagonalization.

    Args: "N" System size
          "H" Hamiltonian
          "to_ord" Dictionary that maps indices from the spin block arrangement
          to that of the tensor product arrangement
          "delta_ts" List/array of delta t
    Returns: "entropy_plot" is a list of values to be plotted.
             "error" is the status of the state choosing function that
             is called from this function. If "error" is True, then no
             state of a zero total <Sz> with an energy density could be found
             for the current configuration.
    """
    spin = 0.5
    points = len(delta_ts) + 1
    entropy_plot = np.zeros(points)

    # Use exact diagonalization for small systems.
    psi_diag, error = common.get_state_blk(H, N)
    H = H.toarray()
    E, V = np.linalg.eigh(H)
    tm = common.TimeMachine(E, V, psi_diag)

    if not error:
        # Plot the rest of the points with time evolution.
        for plot_point in range(points):
            if plot_point == 0:
                psi_diag_evolved = tm.evolve(start_time)
            else:
                psi_diag_evolved = tm.evolve(delta_ts[plot_point - 1])
            psi_ord_evolved = reorder_basis(N, psi_diag_evolved, to_ord)
            entropy_plot[plot_point] += qm.get_vn_entropy(psi_ord_evolved,
                                                          spin, N,
                                                          mode='eqsplit')
    return entropy_plot, error
