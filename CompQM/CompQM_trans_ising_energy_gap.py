from quantum_module import init,sort_eigs
from tran_ising_H import get_tran_ising_H
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import multiprocessing


def get_plot_points(i):
    '''Populates the E_diff list for plotting.'''
    H = get_tran_ising_H(Sx,Sz,b[i],N)
    E,V = eigsh(H,k=4,which='LM')
    E,V = sort_eigs(E,V)
    E_diff = E[1] - E[0]
    return E_diff


S = 1/2                 # Spin
sample_step = 100
N = 15

Sx, Sy, Sz = init(S)

# Multiprocess the plotting process
b = np.linspace(-3.5,3.5,sample_step)
E_diff_pos = list(range(sample_step))
pool = multiprocessing.Pool(processes=4)
E_diff = pool.map(get_plot_points,E_diff_pos)
pool.close()

plt.plot(b,E_diff)
plt.show()
