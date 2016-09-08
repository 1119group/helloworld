import quantum_module as qm
import aubry_andre_block_H as aubryH
import aubry_andre_common as aubryC
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def fluctuation(psi, S):
    fluc = qm.exp_value(S**2, psi) + qm.exp_value(S, psi)**2
    return fluc


def plot_linear_time_evolution(N, h, c, phi, start_time, end_time, n_points):
    # Generate the Hamiltonian and a product state.
    H = aubryH.blk_full(N, h, c, 0, phi)
    psi, error = aubryC.get_state_blk(H, N)
    H = H.tocsc()
    psi = psi.tocsc()
    fluc_plot = []

    if not error:
        # Generate important constants for the next step.
        shift = 0
        for i in range(N, N // 2, -1):
            shift += int(round(sp.misc.comb(N, i)))
        blk_sz = int(round(sp.misc.comb(N, N//2)))

        # Generate the total Sz operator.
        Sx, Sy, Sz = qm.init(0.5)
        Sz_tot = sp.sparse.lil_matrix((2**N, 2**N), dtype=complex)
        for k in range(int(round(N // 2))):
            Sz_tot += qm.get_full_matrix(Sz, k, N)
        Sz_tot = aubryC.Sz2spin_basis(N, Sz_tot)    # Rewrite in the spin basis
        Sz_tot = Sz_tot[shift: shift + blk_sz, shift: shift + blk_sz]   # Slice the center blk

        # Plot the first point.
        U_0 = sp.linalg.expm(-1j * H * start_time)
        psi_0 = U_0 * psi
        fluc = fluctuation(psi_0, Sz_tot)
        fluc_plot.append(fluc)

        # Plot the rest of the points.
        dt = (end_time - start_time) / (n_points - 1)
        U = sp.linalg.expm(-1j * H * dt)

        for p in range(1, n_points):
            psi = U * psi
            fluc = fluctuation(psi, Sz_tot)
            fluc_plot.append(fluc)

    return fluc_plot, error


N = 10
h = 2
c = np.sqrt(2)
phi = 0
start_time = 0
end_time = 20
n_points = 41

fluc_plot, error = plot_linear_time_evolution(N, h, c, phi, start_time, end_time, n_points)
t = np.linspace(start_time, end_time, n_points)

plt.plot(t, fluc_plot)
plt.show()
