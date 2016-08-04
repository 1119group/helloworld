from quantum_module import init,get_full_matrix,sort_eigs
from ising_XY_H import get_ising_XY_H
import numpy as np
from numpy.linalg import eigh
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import multiprocessing

def get_exp_values(b):
    '''Find the expectation value of the spin for a given b.'''
    H = get_ising_XY_H(Sx,Sy,Sz,b,N)
    E,V = eigsh(H,k=4,which='LM')
    E,V = sort_eigs(E,V)
    S__k = get_full_matrix(Sz,3,N)
    
    # Needed to avoid TypeError when converting to dok_matrix.
    psi = np.matrix(V[:,0])
    exp_value = dok_matrix(psi.conj()).dot(S__k.dot(dok_matrix(psi.T)))
    return exp_value[0,0]


S = 1/2                 # Spin
sample_step = 100
N = 20
Sx, Sy, Sz = init(S)
#b = np.linspace(-3,3,sample_step)
#exp_values = []
#for i in b:
    #exp_values.append(get_exp_values(i))

# Parallelize the plotting process
b = np.linspace(-3,3,sample_step)
pool = multiprocessing.Pool(processes=4)
exp_values = pool.map(get_exp_values,b)
pool.close()

plt.plot(b,exp_values)
plt.show()
