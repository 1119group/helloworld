'''
This module provides common functions for MBL localization problems.

7-17-2016
'''

from quantum_module import get_vn_entropy,change_basis,next_permutation
from heisenberg_chain_H import get_heisenberg_chain_H
from block_diagonalized_heisenberg_chain_H import get_block_heisenberg_chain_H
import numpy as np
from numpy.linalg import eigh
from scipy.sparse import dok_matrix,csr_matrix,lil_matrix
from scipy.sparse.linalg import expm_multiply,eigsh
from scipy.misc import comb
from itertools import permutations
import builtins as bi

def entropy_diag(spin,N,E,psi_0,eigvects,t):
    '''Using diagonalized Hamiltonian.'''
    D = int(2*spin+1)**N
    psi_0 = change_basis(psi_0,eigvects)
    psi = np.zeros([D,1],complex)
    for i in range(np.shape(psi_0)[0]):
        psi += psi_0[i,0]*np.exp(-1j*E[i]*t)*np.matrix(eigvects[:,i]).T
    entropy = get_vn_entropy(psi,spin,N,mode='eqsplit')
    return entropy 

def entropy_exp(spin,N,psi_0,H,t):
    '''Using exponentiated Hamiltonian.'''
    psi = expm_multiply(-1j*H*t,psi_0)
    entropy = get_vn_entropy(psi,spin,N,mode='eqsplit')
    return entropy

def get_init_delta_t(time_range_lower_lim,time_range_upper_lim,sample_size):
    log_scale_interval = np.log10(time_range_upper_lim/time_range_lower_lim)
    t_2 = time_range_lower_lim*10**(log_scale_interval/(sample_size-1))
    init_delta_t = t_2 - time_range_lower_lim
    r = t_2/time_range_lower_lim
    return init_delta_t,r
 
def get_delta_delta_t(time_range_lower_lim,plot_point,r):
    '''
    Finds the change of delta_t at each plot point. Only works well
    for plot_point >= 2.
    '''
    delta_delta_t = time_range_lower_lim*r**(plot_point-1)*(r-1)**2
    return delta_delta_t

def get_random_state(Sx,Sy,Sz,spin,N,h,mode='diag',seed=False):
    D = int(2*spin+1)**N
    j = 0
    redo = True
    while redo:
        if seed:
            H = get_heisenberg_chain_H(Sx,Sy,Sz,N,h,seed)
        else:
            H = get_heisenberg_chain_H(Sx,Sy,Sz,N,h)

        if mode == 'expm':
            E_max,eigvects_max = eigsh(H,k=1,which='LA',maxiter=1e6)
            E_min,eigvects_min = eigsh(H,k=1,which='SA',maxiter=1e6)
            E = np.append(E_min,E_max)
        if mode == 'diag':
            E,bi.eigvects = eigh(H.toarray())

        # Create initial state.
        counter = 0
        while True:
            counter += 1
            # Create random psi with magnetization of 0. Here we first form 
            #  a random binary number which has an equal number of 1's and 
            #  0's.
            index = np.zeros(N)
            for k in range(N//2):
                index[k] = 1
            index = np.random.permutation(index)
            # Then we convert the binary number into decimal and put a 1
            #  at the spot indicated by the binary into a zero vector. That
            #  represents a state with an equal number of up and down spins
            #  -- zero magnetization.
            s = 0
            for k in range(N):
                s += int(index[k] * 2**k)
            psi_0 = dok_matrix((D,1),complex)
            psi_0[s,0] = 1
            # Make sure psi's energy density is very close to 0.5.
            exp_val = psi_0.conjtransp().dot(H.dot(psi_0))
            e = (exp_val[0,0] - E[0]) / (E[-1] - E[0])
            if abs(e-0.5) < 0.001:
                redo = False
                j += 1
                break
            # Regenerate the Hamiltonian after failing to generate any state
            #  for a number of times.
            elif counter > D:
                j += 1
                break
    return H, E, psi_0

def get_random_state_Sz_0(Sx,Sy,Sz,spin,N,h):
    D = int(2*spin+1)**N
    j = 0
    redo = True
    while redo:
        s = np.random.random_integers(1e9,size=(10,))
        H_block = get_block_heisenberg_chain_H(Sx,Sy,Sz,N,h,seed=s)
        H_norm = get_heisenberg_chain_H(Sx,Sy,Sz,N,h,seed=s) 
        
        E_max,eigvects_max = eigsh(H_norm,k=1,which='LA',maxiter=1e6)
        E_min,eigvects_min = eigsh(H_norm,k=1,which='SA',maxiter=1e6)
        E = np.append(E_min,E_max)
        
        # Create initial state.
        counter = 0
        while True:
            counter += 1
            # Generate a random state psi with total <Sz> = 0.
            center_block_size = int(comb(N,N/2))
            psi = lil_matrix((D,1),dtype=complex)
            rand_1 = np.random.randint(0,center_block_size)
            psi[int(0.5*(D-center_block_size))+rand_1,0] = 1

            # Make sure psi's energy density is very close to 0.5.
            exp_val = psi.transpose().dot(H_block.dot(psi))
            e = (np.real(exp_val[0,0]) - E[0]) / (E[-1] - E[0])
            if abs(e-0.5) < 0.001:
                redo = False
                j += 1
                break
            # Regenerate the Hamiltonian after failing to generate any state
            #  for a number of times.
            elif counter > 2*center_block_size:
                j += 1
                break
    return H_block,psi
