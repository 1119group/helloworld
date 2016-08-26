import numpy as np
import scipy.sparse as ss
import scipy as sp
import quantum_module as qm
import aubry_andre_block_H as aubh
import aubry_andre_common as aubc
from timer import Timer


def get_ground_state(N, h, c, phi):
    E = []
    V = []
    H_blks = []
    j_max = int(round(0.5 * N))
    # Insert the eigenenergy and eigenstate for very first "block" of H
    H_first = aubh.blk_diag(N, h, c, j_max, phi)
    H_blks.append(H_first.copy())
    E.append(H_first[0, 0])
    V_first = ss.lil_matrix((1, 1))
    V_first[0, 0] = 1
    V.append(V_first.tocsc())

    for j in range(j_max - 1, -1 * j_max, -1):
        H_j = aubh.blk_full(N, h, c, j, phi)
        H_blks.append(H_j.copy())

        E_j_0, V_j_0 = ss.linalg.eigsh(H_j, k=1, which='SA', maxiter=1e6)
        E.append(E_j_0[0])
        V.append(ss.csc_matrix(V_j_0))

    # Insert the eigenenergy and eigenstate for very last "block" of H
    H_last = aubh.blk_diag(N, h, c, -1 * j_max, phi)
    H_blks.append(H_last.copy())
    E.append(H_last[0, 0])
    V_last = ss.lil_matrix((1, 1))
    V_last[0, 0] = 1
    V.append(V_last.tocsc())

    E_min = sorted(E)[0]
    j_rev = E.index(E_min)
    psi_0 = V[j_rev]
    total_Sz = N + 1 - j_rev - j_max
    return H_blks[j_rev], E_min, psi_0, total_Sz


def temporal_correlation(psi_0, U_t, t, S_k, E_0):
    psia = S_k * psi_0
    temp_corr = np.exp(1j * E_0 * t) * psia.transpose().conjugate() * U_t * psia
    temp_corr = np.real(temp_corr[0, 0])
    return float(temp_corr)


def plot_linear_time_evolution(N, h, c, phi, k, start_time, end_time, n_points):
    timer = Timer(n_points, jname='temporal correlation:')
    plot = []
    H, E_0, psi_0, total_Sz = get_ground_state(N, h, c, phi)
    H = H.tocsc()
    psi_0 = psi_0.tocsc()
    blk_sz = H.get_shape()[0]

    j_max = int(round(0.5 * N))
    n = j_max + total_Sz
    shift = 0
    for i in range(N, n, -1):
        shift += int(round(sp.misc.comb(N, i)))

    Sx, Sy, Sz = qm.init(0.5)
    S_k = qm.get_full_matrix(Sz, k, N)
    S_k = aubc.Sz2spin_basis(N, S_k).tolil()
    S_k = S_k[shift:shift + blk_sz, shift:shift + blk_sz].tocsc()

    # Evolve the unitary operator to start_time and plot the first point.
    U_t = ss.linalg.expm(-1j * H * start_time)
    temp_corr = temporal_correlation(psi_0, U_t, start_time, S_k, E_0)
    plot.append(temp_corr)
    timer.progress()

    dt = (end_time - start_time) / (n_points - 1)
    U_t_const = ss.linalg.expm(-1j * H * dt)
    t = np.linspace(start_time, end_time, n_points)
    for i in range(1, n_points):
        U_t = U_t_const * U_t
        temp_corr = temporal_correlation(psi_0, U_t, t[i], S_k, E_0)
        plot.append(temp_corr)
        timer.progress()

    return plot, t


# def plot_log_time_evolution():
