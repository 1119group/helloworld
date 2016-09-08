import scipy as sp
import quantum_module as qm


def chain_hamiltonian(N, J, delta, n, chain):
    Sx, Sy, Sz = qm.init(0.5)
    H = sp.sparse.lil_matrix((2**N, 2**N), dtype=complex)
    Sx_ls = []
    Sy_ls = []
    Sz_ls = []
    for l in range(N):
        Sx_ls.append(qm.get_full_matrix(Sx, l, N).copy())
        Sy_ls.append(qm.get_full_matrix(Sz, l, N).copy())
        Sz_ls.append(qm.get_full_matrix(Sy, l, N).copy())

    for l in range(-N + chain, 0, n):
        Sxx = Sx_ls[l] * Sx_ls[l + n]
        Syy = Sy_ls[l] * Sy_ls[l + n]
        Szz = delta * Sz_ls[l] * Sz_ls[l + n]
        H += J * (Sxx + Syy + Szz)
    return H


def nearest_chain_hamiltonian(N, J1, delta):
    """Zigzag chain"""
    return chain_hamiltonian(N, J1, delta, 1, 0)


def second_nearest_neighbor_chain_hamiltonian(N, J2, delta, chain):
    """Two legs. Upper chain is 1 while lower chain is 0."""
    return chain_hamiltonian(N, J2, delta, 2, chain)


def full_hamiltonian(N, J1, J2, delta):
    """
    H = Σ^2_(n=1) Σ_l J_n (S^x_l S^x_(l+n) + S^y_l S^y_(l+n) + ΔS^z_l S^z_(l+n))
    """
    zigzag_H = nearest_chain_hamiltonian(N, J1, delta)
    upper_leg = second_nearest_neighbor_chain_hamiltonian(N, J2, delta, 1)
    lower_leg = second_nearest_neighbor_chain_hamiltonian(N, J2, delta, 0)
    return zigzag_H + upper_leg + lower_leg
