from quantum_module import init,sort_eigs
from ising_XY_H import get_ising_XY_H
import numpy as np
from numpy.linalg import eigh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
#from multiprocessing import Pool
from joblib import Parallel,delayed

def get_E_diff(b):
    H = get_ising_XY_H(Sx,Sy,Sz,b,N)
    E,V = eigsh(H,k=4,which='LM')
    E,V = sort_eigs(E,V)
    E_diff = E[1] - E[0]
    return E_diff

S = 1/2                 # Spin
sample_size = 200
b = np.linspace(-3.5,3.5,sample_size)    # Coefficient in the XY Hamiltonian.
N = 12

Sx, Sy, Sz = init(S)

#E_diff = np.empty(sample_size)
#for i in range(sample_size):
    #H = get_ising_XY_H(Sx,Sy,Sz,b[i],N)
    #E,V = eigsh(H,k=4,which='LM')
    #E,V = sort_eigs(E,V)
    #E_diff[i] = E[1] - E[0]
    

#pool = Pool(processes=4)
#E_diff = pool.map(get_E_diff,b)
#pool.close()

#E_diff = Parallel(n_jobs=4)(delayed(get_E_diff)(b[i]) 
    #for i in range(sample_size))
    
with Parallel(n_jobs=4) as parallel:
    E_diff = parallel(delayed(get_E_diff)(b[i]) 
        for i in range(sample_size))
        
plt.plot(b,E_diff)
plt.show()
#plt.savefig("QM_3.17_phase_trans_20spins.png")
