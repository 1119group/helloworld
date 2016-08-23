"""
This module provides common functions to do quantum mechanical calculations.

8-9-2016
"""

import numpy as np
from numpy.random import random_sample, randint
from numpy.linalg import eigvalsh
from scipy import io
from scipy.sparse import dok_matrix, lil_matrix, eye, kron, issparse
from time import time
import sys
import os


def init(S):
    """
    Initializes calculations by computing Sx, Sy and Sz. All calculations
    are done in the z-basis.
    Returns sparse matrices.
    """
    # Create the raising and lowering operators.
    S_plus = dok_matrix((int(2 * S + 1), int(2 * S + 1)), float)
    for i in np.arange(-S, S, 1):
        m = -i - 1
        S_plus[int(i + S), int(i + S + 1)] = np.sqrt(S * (S + 1) - m * (m + 1))
    S_minus = S_plus.transpose()

    # Create Sx, Sy and Sz.
    Sx = 0.5 * (S_plus + S_minus)
    Sy = -0.5j * (S_plus - S_minus)
    Sz = dok_matrix((int(2 * S + 1), int(2 * S + 1)), float)
    for i in np.arange(S, -S - 1, -1):
        Sz[-int(i - S), -int(i - S)] = i

    return Sx, Sy, Sz


def iowrapper(func, fname, options):
    """
    Wrapper for I/O operations to enable loading and dumping cache to disk.
    The data in question must be a scipy sparse matrix for this wrapper
    to work properly.

    Args: "func" is a function to be wrapped such that it could have the
          added ability to dump cache to disk.
          "fname" is the name of the cache file to be dumped/loaded to/from
          disk.
          "options" is a **list** of arguments (literally arguments enclosed
          in a Python list) to be passed on to "func." "func" must be written
          such that it has the ability to unpack arguments from a list.
    Return: An operator/matrix which "func" is able to independently return,
            only now it could be loaded from disk instead of being
            regenerated from scratch every single time.
    """
    if os.path.isfile(fname):
        operator = io.loadmat(fname)['i']
    else:
        operator = func(options)
        io.savemat(fname, {'i': operator}, appendmat=False)
    return operator


def sp_den_col_row_compat(N, vec_in, func, options):
    """
    This wrapper function provides compatibility for functions working with
    vectors with sparse, dense, column and row vectors as long as the
    function in question is written to work with both sparse and dense
    column vectors.

    Args: "vec_in" is the vector to be passed on to the function for
          processing. It could be sparse, dense, column or row, as long as
          it is "two dimensional" meaning that it must have a shape of
          (k, 1) or (1, k) but not (k, ).
          "func" is the function to be wrapped. It must have the ability to
          unpack its options from a Python list. All in all, it must take
          in the following arguments:
              1. "N", the size of the particle system
              2. "vec_in", the column vector to be processed
              3. "vec_out", a zero vector with dimensions corresponding to
                 a full vector of a size N spin 1/2 particle system that
                 "func" will return eventually
              4. "options", a Python list containing all other options.
          "options" is a list of options enclosed in a Python list.
    Returns: A vector which attributes mirrors that of the vector passed
             into this function.
    """
    D = 2 ** N

    # Check sparsity of the vector.
    if issparse(vec_in):
        vdim = vec_in.get_shape()[0]
        # Convert the vector into a column if it is not one already.
        if vdim == 1:
            vec_in = vec_in.transpose().conjugate()
        vec_in = vec_in.tolil()
        vec_out = lil_matrix((D, 1), dtype=complex)
    else:
        vdim = np.shape(vec_in)[0]
        # Convert the vector into a column if it not one already.
        if vdim == 1:
            vec_in = vec_in.T.conjugate()
        vec_out = np.zeros([D, 1], dtype=complex)

    # Call the actual function.
    vec_out = func(N, vec_in, vec_out, options)

    # Convert the processed vector back to a row vector if it was one
    #  to begin with.
    if issparse(vec_in):
        if vdim == 1:
            vec_out = vec_out.transpose().conjugate()
    else:
        if vdim == 1:
            vec_out = vec_out.T.conjugate()

    return vec_out


def get_full_matrix(S, k, N):
    """
    Builds the S matrices in an N particle system. Assumes periodic boundary
    condition.
    "S" could be an operator/state we want to work on. If it is a state, it
    must be put in a column vector form. "S" must be sparse.
    "k" is the location index of the particle in a particle chain. The first
    particle has k=0, the second has k=1 and so on.
    Returns a sparse matrix.
    """
    D = S.get_shape()[0]  # Dimensions of the operator/state vector.

    if k == 0 or k == N:  # k = N is the periodic boundary condition.
        S_full = dok_matrix(S)
        S_full = kron(S_full, eye(D ** (N - 1)))
    elif k == 1:
        S_full = eye(D)
        S_full = kron(S_full, S)
        S_full = kron(S_full, eye(D ** (N - 2)))
    else:
        S_full = eye(D)
        S_full = kron(S_full, eye(D ** (k - 1)))
        S_full = kron(S_full, S)
        S_full = kron(S_full, eye(D ** (N - k - 1)))

    return S_full.tocsr()


def exp_value(S, psi):
    """
    Computes the expected value of an operator with a given state "psi."
    "S" is an operator/observable and must be a sparse matrix.
     "psi" is a column vector and its sparsity is optional.
    """
    if not issparse(psi):
        psi = dok_matrix(np.matrix(psi))
    if np.shape(psi)[1] != 1:
        psi = psi.conjugate().transpose()

    exp_val = psi.transpose().conjugate().dot(S.dot(psi))
    exp_value = np.real(exp_val[0, 0])
    return float(exp_value)


def sort_eigs(E, V):
    """
    Sort the given set of eigenvectors and eigenvectors by the eigenvalues
    from the least to the greatest.
    "E" is a list of eigenvalues.
    "V" is an eigenvector matrix/array.
    """
    temp = []
    E_sorted = np.copy(E)
    V_sorted = np.copy(V)
    for j in range(np.shape(E)[0] - 1):
        for i in range(np.shape(E)[0] - 1):
            if E_sorted[i] > E_sorted[i + 1]:
                temp.append(np.copy(V_sorted[:, i]))
                V_sorted[:, i] = V_sorted[:, i + 1]
                V_sorted[:, i + 1] = temp.pop()
                temp.append(E_sorted[i])
                E_sorted[i] = E_sorted[i + 1]
                E_sorted[i + 1] = temp.pop()
    return E_sorted, V_sorted


def red_rho_A_1spin(psi, spin):
    """
    Forms a reduced ground state density matrix from a given state "psi."
    Every particle aside from the first will be traced out.
    "psi" must be a column vector. Sparsity is optional. "spin" is the spin
    number.
    Returns a sparse matrix.
    """

    dim = int(2 * spin + 1)  # Dimensions of the spin matrices.

    # Accommodate both sparse and dense matrices for compatibility.
    if issparse(psi):
        psi = psi.todense()

    psi_reshaped = np.reshape(psi, (dim, int(np.shape(psi)[0] / 2)))
    rho_A = np.dot(psi_reshaped, np.conjugate(np.transpose(psi_reshaped)))
    return dok_matrix(rho_A)


def red_rho_eqsplit(psi, spin, N):
    """
    Forms a reduced density matrix from a given state "psi" for system AB
    in which the size of A equals that of B. "N" must be an even number.
    "psi" must be a column vector. Sparsity is optional. "spin" is the spin
    number.
    Returns a sparse matrix.
    """
    if N % 2 != 0:
        raise Exception("N must be an even integer.")

    dim = int(2 * spin + 1) ** int(N / 2)  # Dimensions of the spin matrices.

    # Accommodate both sparse and dense matrices for compatibility.
    if issparse(psi):
        psi = psi.todense()

    psi_reshaped = np.reshape(psi, (dim, dim))
    rho_A = np.dot(psi_reshaped, np.conjugate(np.transpose(psi_reshaped)))
    return dok_matrix(rho_A)


def get_vn_entropy(psi, spin, N=None, mode='1spin', base='e'):
    """
    Compute the von Neumann entropy for a given state "psi".
    "psi" must be a column vector. Sparsity is optional.
    Available modes: "1spin" for the entanglement entropy for only the first
    spin, "eqsplit" for the entanglement entropy for an evenly split system.
    Available bases: "e" for natural log, "2" for log2 and "10" for log10.
    """
    if N is None and mode == 'eqsplit':
        raise Exception("N cannot be 'None' for mode='eqsplit'.")

    if mode == '1spin':
        red_rho_A = red_rho_A_1spin(psi, spin)
    elif mode == 'eqsplit':
        red_rho_A = red_rho_eqsplit(psi, spin, N)

    lamb = eigvalsh(red_rho_A.todense())  # Eigenvalues of the reduced matrix.
    S_AB_terms = []
    for i in range(np.shape(red_rho_A)[0]):
        if abs(lamb[i]) < 1e-6:
            # lim a->0 (alog(a)) = 0. It also removes some minuscule negative
            #  lambda values resulting from rounding errors.
            S_AB_terms.append(0)
        else:
            if base == 'e':
                S_AB_terms.append(-lamb[i] * np.log(lamb[i]))
            elif base == '2' or base == 2:
                S_AB_terms.append(-lamb[i] * np.log2(lamb[i]))
            elif base == '10' or base == 10:
                S_AB_terms.append(-lamb[i] * np.log10(lamb[i]))
            else:
                raise Exception('Available bases are "e", "2" and "10"')

    return np.sum(S_AB_terms)


def get_init_delta_t(time_range_lower_lim, time_range_upper_lim, sample_size):
    """
    This function provides the initial time step for a logarithmic time
    axis.
    """
    log_scale_interval = np.log10(time_range_upper_lim / time_range_lower_lim)
    t_2 = time_range_lower_lim * 10 ** (log_scale_interval / (sample_size - 1))
    init_delta_t = t_2 - time_range_lower_lim
    r = t_2 / time_range_lower_lim
    return init_delta_t, r


def get_delta_delta_t(time_range_lower_lim, plot_point, r):
    """
    Finds the change of delta_t at each plot point for a logarithmic time
    axis. Only works well for plot_point >= 2.
    """
    delta_delta_t = time_range_lower_lim * r ** (plot_point - 1) * (r - 1) ** 2
    return delta_delta_t


def get_log_t_axis(time_range_lower_lim, time_range_upper_lim, sample_size):
    """
    Sets up the time axis for a logarithmic plot. Returns a list of
    plot points.
    """
    t = []
    t.append(time_range_lower_lim)
    current_t = time_range_lower_lim
    init_delta_t, r = get_init_delta_t(time_range_lower_lim,
                                       time_range_upper_lim, sample_size)
    current_delta_t = init_delta_t
    t.append(time_range_lower_lim + current_delta_t)
    for plot_point in range(2, sample_size):
        delta_delta_t = get_delta_delta_t(time_range_lower_lim, plot_point, r)
        current_delta_t += delta_delta_t
        current_t += current_delta_t
        t.append(current_t)
    return t


def rand_sign():
    """Returns a random positive or negative sign."""
    s = randint(0, 2)
    if s == 1:
        sign = 1
    elif s == 0:
        sign = -1
    return sign


def rand_cnum():
    """Returns a random complex number with a modulus in [0,1]."""
    sign_a = rand_sign()
    sign_b = rand_sign()
    cnum = sign_a * random_sample() / 2 + sign_b * random_sample() * 0.5j
    return cnum


def change_basis(S, basis):
    """
    Project a given state/operator "S" onto a new basis.
    "S" is a given sparse matrix or a sparse column vector.
    "basis" is a complete set of orthonormal vectors, ordered in a
    matrix with each column vector being a basis vector.
    Returns a sprase column vector.
    """
    if issparse(basis) is False:  # basis sparsity is optional.
        basis = dok_matrix(basis)

    if np.shape(S)[0] == np.shape(S)[1]:  # For operators
        U = dok_matrix(basis)
        S_nb = U.conjtransp().dot(S.dot(U))
    else:  # For vectors
        dim = int(np.shape(basis)[1])
        S_nb = dok_matrix((dim, 1), dtype=complex)
        for i in range(dim):
            S_nb[i, 0] = S.transpose().conjugate().dot(basis[:, i])[0, 0]
    return S_nb


def next_permutation(l):
    """
    Code plagiarized from StackOverflow. With a given list of values,
    this function changes the list in situ to the next permutation.
    This function differs from itertools.permutations in that it takes
    into account repeated values in the list and avoids returning duplicates.

    "l" is a list which this function will modify.
    """
    n = len(l)
    # Step 1: Find tail
    last = n - 1  # tail is from `last` to end
    while last > 0:
        if l[last - 1] < l[last]:
            break
        last -= 1
    # Step 2: Increase the number just before tail
    if last > 0:
        small = l[last - 1]
        big = n - 1
        while l[big] <= small:
            big -= 1
        l[last - 1], l[big] = l[big], small
    # Step 3: Reverse tail
    i = last
    j = n - 1
    while i < j:
        l[i], l[j] = l[j], l[i]
        i += 1
        j -= 1
    return l


def permute_one_zero(l):
    """
    Find the next permutation of a list of ones and zeros. This function
    permutes in the reverse order of next_permutation.

    "l" is a list which this function will modify.
    """
    n = len(l) - 1
    migrate = False
    while True:
        # Find the last 1
        i = n
        while True:
            if l[i] == 1:
                break
            else:
                i -= 1
        # Switch the element with the next element if the element is
        #  not the last element.
        if i != n:
            l[i], l[i + 1] = l[i + 1], l[i]
            if migrate:
                i += 2
                j = i
                # Find the first 1 to the right of the 1 we just moved.
                while True:
                    if l[j] == 1:
                        break
                    else:
                        j += 1
                        if j >= len(l):
                            break
                # Move all the 1's to the left.
                w = len(l[j:])
                for k in range(w):
                    l[i], l[j] = l[j], l[i]
                    i += 1
                    j += 1
                migrate = False
            break
        # Since there is a 1/some 1's at the very end of the list,
        #  we loop to look for the next one to the left that is
        #  separated by some 0's.
        else:
            # A flag to tell the function to move all the 1's at
            #  the right end to the left.
            migrate = True
            n -= 1
    return l


def sec_to_human_readable_format(time):
    """
    Converts time (in seconds) into the HHMMSS format. Returns a string.
    """
    Days = str(int(time // 86400))
    Hr = int((time % 86400) // 3600)
    if Hr < 10:
        Hr = "0" + str(Hr)
    else:
        Hr = str(Hr)
    Min = int((time % 3600) // 60)
    if Min < 10:
        Min = "0" + str(Min)
    else:
        Min = str(Min)
    Sec = round(time % 60)
    if Sec < 10:
        Sec = "0" + str(Sec)
    else:
        Sec = str(Sec)
    if Days == str(0):
        elapsed_time = Hr + ":" + Min + ":" + Sec + "        "
    elif Days == str(1):
        elapsed_time = Days + " Day " + Hr + ":" + Min + ":" + Sec + "  "
    else:
        elapsed_time = Days + " Days " + Hr + ":" + Min + ":" + Sec
    return elapsed_time


# The following functions are DEPRECATED and contain unfixed bugs.
#  Do not use. Use the Timer class instead.

def show_progress(start_time, iteration, total, barLength=25):
    """
    Prints the progress on screen. "start_time" is the the start time of the
    entire program. "iteration" is the job number of the current job. As with
    everything in Python, it is 0 based. For instance, if the current job is
    the first task, iteration=0. "total" is the total number of tasks to
    perform.
    The output is in the hh:mm:ss format.
    """
    # Calculate time used for progress report purposes.
    elapsed = time() - start_time
    ET_sec = elapsed / (iteration + 1) * (total - iteration - 1)
    ET = sec_to_human_readable_format(ET_sec)

    if iteration == 0:
        report_time = ""
        filledLength = 0
        percent = 0
    else:
        report_time = "Est. time: " + ET
        filledLength = int(round(barLength * (iteration + 1) / total))
        percent = round(100.00 * ((iteration + 1) / total), 1)

    bar = '\u2588' * filledLength + '\u00B7' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % ('Progress:', bar,
                                            percent, '%  ', report_time)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


def show_elapsed_time(start_time):
    """Prints the total elapsed time."""
    elapsed = time() - start_time
    elapsed_time = sec_to_human_readable_format(elapsed)
    print("\nTime elapsed: " + elapsed_time)
