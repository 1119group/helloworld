from quantum_module import init,get_full_matrix
from ising_XY_H import get_ising_XY_H
import numpy as np
from numpy.linalg import eigh
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt
import multiprocessing

def get_exp_value(t):
    """
    Get the expectation value of the state psi for Sx, Sy and Sz with respect
    to time t.
    The state is defined as:
    psi_0 = 1/sqrt(2)*(psi_0 + psi_1)
    """
    psi = 1/np.sqrt(2)*np.exp(-1j*E[0]*t)*eigvects[:,0]
    psi += 1/np.sqrt(2)*np.exp(-1j*E[1]*t)*eigvects[:,1]
    
    #S_T = dok_matrix((D**N,D**N))
    #for k in range(N):
        #S_k_full = get_full_matrix(Sn,k,N)
        #S_T += S_k_full
    S_T = get_full_matrix(Sn,0,N)
    exp_val = np.dot(np.dot(np.conj(psi),S_T.toarray()),np.transpose(psi))
    return exp_val
    
    
S = 1/2                 # Spin
sample_step = 300
b = 8
N = 10
Sx, Sy, Sz = init(S)
D = Sx.get_shape()[0]
H = get_ising_XY_H(Sx,Sy,Sz,b,N)
E,eigvects = eigh(H.toarray())
time = np.linspace(0,10,sample_step)

for Sn in [Sx,Sy,Sz]:
    pool = multiprocessing.Pool(processes=4)
    exp_val = pool.map(get_exp_value,time)
    pool.close()
    plt.plot(time,exp_val)

plt.show()
