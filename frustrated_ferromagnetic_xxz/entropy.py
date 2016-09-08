import numpy as np
from scipy.sparse.linalg import eigsh
import quantum_module as qm
import hamiltonian as hm
from timer import Timer
import matplotlib.pyplot as plt


def ground_state(N, J1, J2, delta):
    H = hm.full_hamiltonian(N, J1, J2, delta)
    E_0, psi_0 = eigsh(H, k=1, which='SA', maxiter=1e6)
    return psi_0


def ground_state_entropy(N, J1, J2, delta):
    psi_0 = ground_state(N, J1, J2, delta)
    ent = qm.get_vn_entropy(psi_0, 0.5, N, mode='eqsplit')
    return ent


def plot_entropy_delta(N, J1, J2, deltas):
    ent_plot = []
    for delta in deltas:
        ent_plot.append(ground_state_entropy(N, J1, J2, delta))
        timer.progress()
    return ent_plot


def plot_entropy_J1J2(N, J1, frac_J2J1s, delta):
    ent_plot = []
    for frac_J2J1 in frac_J2J1s:
        J2 = J1 * frac_J2J1
        ent_plot.append(ground_state_entropy(N, J1, J2, delta))
        timer.progress()
    return ent_plot


N = 10
sample_size = 20
frac_J2J1 = -1 / 2
frac_J2J1s = np.linspace(-3.4, -3, sample_size)
J1 = 1
J2 = J1 * frac_J2J1
delta = 0.8
deltas = np.linspace(0.5, 1, sample_size)

timer = Timer(sample_size)
# plot = plot_entropy_delta(N, J1, J2, deltas)
plot = plot_entropy_J1J2(N, J1, frac_J2J1s, delta)

# plt.ylim(0, 3)
# plt.plot(deltas, plot, marker='o', markersize=4.5)
plt.plot(frac_J2J1s, plot, marker='o', markersize=4.5)
plt.show()
