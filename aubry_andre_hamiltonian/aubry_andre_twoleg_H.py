import quantum_module as qm
import numpy as np


def full_smatrices(N):
    spin = 0.5
    Sx, Sy, Sz = qm.init(spin)
    Sxs, Sys, Szs = [], [], []
    for k in range(N):
        Sxs.append(qm.get_full_matrix(Sx, k, N))
        Sys.append(qm.get_full_matrix(Sy, k, N))
        Szs.append(qm.get_full_matrix(Sz, k, N))
    return Sxs, Sys, Szs


def nearest_neighbor_interation(N, I):
    total_interaction = []
    for k in range(N):
        total_interaction.append(Sxs[k - I] * Sxs[k])
        total_interaction.append(Sys[k - I] * Sys[k])
        total_interaction.append(Szs[k - I] * Szs[k])
    return sum(total_interaction)


def inter_chain_interation(N, I):
    total_interaction = []
    for k in range(N):
        if not (k + 1) % I == 0:
            total_interaction.append(Sxs[k] * Sxs[k + 1])
            total_interaction.append(Sys[k] * Sys[k + 1])
            total_interaction.append(Szs[k] * Szs[k + 1])
    return sum(total_interaction)


def field_terms(N, h, c, phi, I):
    sites = np.array(range(1, N // I + 1)).repeat(I, axis=0)
    field = h * np.cos(2 * np.pi * c * sites + phi)
    for i, Sz in enumerate(Szs):
        Sz *= field[i]
    return sum(Szs)


def full_hamiltonian(N, h, c, phi, J1, J2, I):
    global Sxs, Sys, Szs
    Sxs, Sys, Szs = full_smatrices(N)
    neighor_terms = nearest_neighbor_interation(N, I)
    inter_leg_terms = inter_chain_interation(N, I)
    field_contribution = field_terms(N, h, c, phi, I)
    return J1 * neighor_terms + J2 * inter_leg_terms + field_contribution
