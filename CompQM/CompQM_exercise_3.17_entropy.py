from quantum_module import init,red_rho_A_1spin,sort_eigs
from ising_XY_H import get_ising_XY_H
import numpy as np
from numpy.linalg import eigvalsh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import multiprocessing
    
def get_vn_entropy(b):
    '''Compute the ground state von Neumann entropy for a given b.'''
    H = get_ising_XY_H(Sx,Sy,Sz,b,N)
    E,eigvects = eigsh(H,k=1,which='LM',maxiter=1e6)
    red_rho_A = red_rho_A_1spin(eigvects[:,0],S)
    lamb = eigvalsh(red_rho_A.todense()) # Eigenvalues of the reduced matrix.
    S_AB_terms = []
    for i in range(np.shape(red_rho_A)[0]):
        if abs(lamb[i]) < 1e-6:
            # lim a->0 (alog(a)) = 0. It also removes some minuscule negative 
            #  lambda values resulting from rounding errors.
            S_AB_terms.append(0)
        else:
            S_AB_terms.append(-lamb[i]*np.log2(lamb[i]))
        
    return np.sum(S_AB_terms)


S = 1/2                 # Spin
sample_step = 100
N = 6

Sx, Sy, Sz = init(S)
D = Sx.get_shape()[0]              # Dimensions of the spin matrices.

# Parallelize the plotting process
b = np.linspace(-3,3,sample_step)    
pool = multiprocessing.Pool(processes=4)
entropy = pool.map(get_vn_entropy,b)
pool.close()

plt.plot(b,entropy)
plt.show()
