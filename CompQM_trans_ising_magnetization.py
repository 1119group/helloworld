from quantum_module import init,get_full_matrix,sort_eigs
from tran_ising_H import get_tran_ising_H
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import multiprocessing

def get_exp_values(b):
    '''Find the expectation value of the spin for a given b.'''
    H = get_tran_ising_H(Sx,Sz,b,N)
    E,V = eigsh(H,k=4,which='LM')
    E,V = sort_eigs(E,V)
    S__k = get_full_matrix(Sx,1,N)
    
    # Needed to avoid TypeError when converting to dok_matrix.
    psi = np.matrix(V[:,0])
    exp_value = dok_matrix(psi.conj()).dot(S__k.dot(dok_matrix(psi.T)))
    return exp_value[0,0]


S = 1/2                 # Spin
sample_step = 100
#b = 1
N = 10

Sx, Sy, Sz = init(S)

# Parallelize the plotting process
b = np.linspace(-3,3,sample_step)
pool = multiprocessing.Pool(processes=4)
exp_values = pool.map(get_exp_values,b)
pool.close()

plt.plot(b,exp_values)
plt.show()
