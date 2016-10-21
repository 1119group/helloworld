"""
This function generates a block diagonalized Hamiltonian for a
multi-legged Hamiltonian with a pseudo-random, quasi-periodic field.

10-20-2016
"""

import quantum_module as qm
import numpy as np
import scipy.sparse as sp
from scipy.misc import comb
from scipy import io
import pickle
import functools


def _abs_diff(x, y):
    return abs(x - y)


def cache(function):
    """Generic caching wrapper. Should work on any kind of I/O"""
    @functools.wraps(function)
    def wrapper(*args, **kargs):
        try:
            cachefile = './cache/{}{}'.format(function.__name__, (args, kargs))
            with open(cachefile, 'rb') as c:
                return pickle.load(c)
        except FileNotFoundError:
            result = function(*args, **kargs)
            with open(cachefile, 'wb') as c:
                pickle.dump(result, c, pickle.HIGHEST_PROTOCOL)
            return result
    return wrapper


def cachemat(function):
    """Caching wrapper for sparse matrix generating functions."""
    @functools.wraps(function)
    def wrapper(*args, **kargs):
        try:
            cachefile = './cache/{}{}'.format(function.__name__, (args, kargs))
            return io.loadmat(cachefile)['i']
        except FileNotFoundError:
            result = function(*args, **kargs)
            io.savemat(cachefile, {'i': result}, appendmat=False)
            return result
    return wrapper


def bin_to_dec(l):
    """Converts a list "l" of 1s and 0s into a decimal"""
    return int(''.join(map(str, l)), 2)


@cache
def create_complete_basis(N, current_j):
    """Creates a complete basis for the current total <Sz>"""
    dim = 2 ** N
    spin_ups = round(0.5 * N + current_j)
    spin_downs = N - spin_ups
    blksize = int(round(comb(N, spin_ups)))
    basis_seed = [0] * spin_downs + [1] * spin_ups
    basis = basis_seed
    # "to_diag" is a dict that maps ordinary indices to block diagonalized
    #  indices. "to_ord" is the opposite.
    basis_set, to_diag, to_ord = [], {}, {}
    for i in range(blksize):
        try:
            basis = qm.permute_one_zero(basis)
        except IndexError:                # When current_j is N // 2 or -N // 2
            pass
        basis_set.append(basis[:])
        decimal_basis = bin_to_dec(basis)
        to_diag[decimal_basis] = i      # i is the index within only this block
        to_ord[i] = dim - decimal_basis - 1
    return basis_set, to_diag, to_ord


def diagonal_single_block(N, h, c, phi, J1, J2, I, current_j):
    """
    Creates the diagonal of a block of the Hamiltonian.

    Args: 'N' System length
          'h' Field strength
          'c' Irrational/transcendental number happens to be in the field
          'phi' Phase shift
          'J1' Interaction constant between sites along a leg
          'J2' Interaction constant between legs
          'I' Number of legs
          'current_j' Total <Sz>
    Returns: Sparse matrix (dia matrix)
    """
    basis_set = create_complete_basis(N, current_j)[0]
    blksize = len(basis_set)
    diagonal = np.zeros(blksize)
    sites = np.array(range(1, N // I + 1)).repeat(I, axis=0)
    field = h * np.cos(2 * np.pi * c * sites + phi)
    d = {1: 1, 0: -1}         # For conversion of 0s to -1s in the basis later
    for i, b in enumerate(basis_set):
        # ***Interaction Terms***
        # Number of repeated 1s and 0s separated by I
        #  Compute absolute values of differences of pairs separated by I
        #  and sum. Periodic BC (horizontal interaction).
        diff_pairs = sum(map(_abs_diff, b, b[I:] + b[:I]))
        same_pairs = N - diff_pairs
        diagonal[i] += 0.25 * J1 * (same_pairs - diff_pairs)

        # Number of different adjacent 1s and 0s.
        #  Closed BC (interaction between legs)
        if not I <= 1:
            comp = [m for m in range(N) if not (m + 1) % I == 0]
            diff_pairs = sum(map(_abs_diff, [b[m] for m in comp],
                                 [b[m + 1] for m in comp]))
            same_pairs = len(comp) - diff_pairs
            diagonal[i] += 0.25 * J2 * (same_pairs - diff_pairs)

        # ***Field Terms***
        diagonal[i] += 0.5 * sum([d[m] for m in b] * field)
    return sp.diags(diagonal, 0, dtype=complex)


@cachemat
def off_diagonal_single_block(N, J1, J2, I, current_j):
    """
    Creates the off diagonals of a block of the Hamiltonian.

    Args: 'N' System size
          'J1' Coupling constant between neighboring sites
          'J2' Coupling constant between legs
          'I' Number of legs
          'current_j' Total <Sz>
    Returns: Sparse matrix (lil matrix)
    """
    def non_zero_element(i, bi, pair, J):
        """Sets non-zero elements in the matrix"""
        bj = bi[:]
        bj[pair[0]], bj[pair[1]] = bj[pair[1]], bj[pair[0]]
        if not sum(map(_abs_diff, bi, bj)) == 0:
            j = to_diag[bin_to_dec(bj)]
            off_diagonal[i, j] = 0.5 * J

    basis_set, to_diag, to_ord = create_complete_basis(N, current_j)
    blksize = len(basis_set)
    off_diagonal = sp.lil_matrix((blksize, blksize), dtype=complex)

    # Pairs of elements to inspect. I_pairs = pairs I sites apart
    #  adjacent_pairs = pairs adjacent to each other
    I_pairs = [(N + i if i < 0 else i, i + I) for i in range(-I, N - I)]
    adjacent_pairs = [(i, i + 1) for i in range(N - 1) if not (i + 1) % I == 0]

    for i, bi in enumerate(basis_set):
        # Flipping of elements I sites apart
        for pair in I_pairs:
            non_zero_element(i, bi, pair, J1)

        # Flipping of elements adjacent to each other
        if not I <= 1:
            for pair in adjacent_pairs:
                non_zero_element(i, bi, pair, J2)
    return off_diagonal


@cachemat
def single_block(N, h, c, phi, J1=1, J2=1, I=2, current_j=0):
    """
    Creates a block of the Hamiltonian

    Args: 'N' System length
          'h' Field strength
          'c' Irrational/transcendental number happens to be in the field
          'phi' Phase shift
          'J1' Interaction constant between sites along a leg
          'J2' Interaction constant between legs
          'I' Number of legs
          'current_j' Total <Sz>
    Returns: Sparse matrix (CSC matrix)
    """
    diagonals = diagonal_single_block(N, h, c, phi, J1, J2, I, current_j)
    off_diagonals = off_diagonal_single_block(N, J1, J2, I, current_j)
    return diagonals + off_diagonals
