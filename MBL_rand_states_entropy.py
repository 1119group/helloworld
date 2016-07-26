from quantum_module import init,show_elapsed_time,show_progress
from mbl import entropy_diag,entropy_exp,get_random_state
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import norm
from multiprocessing import Pool
import os
from time import time
import builtins as bi

def get_entropy_diag(t):
    '''Wrapper of entropy_diag() for pool.'''
    entropy = entropy_diag(spin,N,E,psi_0,bi.eigvects,t)
    return entropy

def get_entropy_exp(t):
    '''Using exponentiated Hamiltonian.'''
    entropy = entropy_exp(spin,N,psi_0,H,t)
    return entropy

def plot_rand_states(mode='diag',seed=False):
    global E,psi_0,H
    entropy_sum = np.zeros(sample_size)
    start_time = time()
    H_seeds = list(range(10*psi_count))
    for i in range(psi_count):
        if not seed:
            H, E, psi_0 = get_random_state(Sx,Sy,Sz,spin,N,h,mode)
        else:
            H, E, psi_0 = get_random_state(Sx,Sy,Sz,spin,N,h,mode,H_seeds[i])

        t = np.linspace(0,3300,sample_size)
        pool = Pool(processes=threads)
        if mode == 'diag':
            entropy = pool.map(get_entropy_diag,t)
        elif mode == 'expm':
            entropy = pool.map(get_entropy_exp,t)
        pool.close()
        entropy_sum += entropy
        show_progress(start_time,i,psi_count)

    entropy_sum /= psi_count
    show_elapsed_time(start_time)
    print("Done.")
    return t, entropy_sum

spin = 1/2
N = 8
h = 2
D = int(2*spin+1)**N
Sx,Sy,Sz = init(spin)
psi_count = 20
sample_size = 30               # Graphing sample size.
threads = os.cpu_count()

# Find the average of many initial states.
t,entropy_sum = plot_rand_states(mode='diag',seed=False)

plt.xlim(0.1,1e4)
plt.xlabel('Time (t)')
plt.ylabel('S(t)')
plt.title('MBL entropy spin=10 h=7 avreaged over 200 states')
plt.semilogx(t,entropy_sum,linewidth=0,
           marker='o',markerfacecolor='green', markeredgewidth=0)
plt.savefig('MBL_entropy_10spins_h7_avg200states.png')
