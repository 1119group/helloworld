from quantum_module import init,get_full_matrix,sort_eigs
from ising_XY_H import get_ising_XY_H
from math import floor
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt
import multiprocessing

def get_exp_val(S,psi):
    '''
    Find the expected values for an observable with a given state.
    S must be a sparse matrix.
    '''
    #exp_value = np.dot(np.dot(np.conj(psi),S.toarray()),np.transpose(psi))
    psi = np.matrix(psi)
    exp_value = dok_matrix(psi.conj()).dot(S.dot(dok_matrix(psi.T)))
    return exp_value[0,0]
    #return exp_value

def get_correlation(b):
    '''Find the correlation of a spin pair for a given b.'''
    H = get_ising_XY_H(Sx,Sy,Sz,b,N)
    E,V = eigsh(H,k=4,which='LM')
    E,V = sort_eigs(E,V)
    
    corr = get_exp_val(S_product,V[:,0])
    corr -= get_exp_val(Sx_k,V[:,0])*get_exp_val(Sx_k_delta,V[:,0])
    corr -= get_exp_val(Sy_k,V[:,0])*get_exp_val(Sy_k_delta,V[:,0])
    corr -= get_exp_val(Sz_k,V[:,0])*get_exp_val(Sz_k_delta,V[:,0])
    return corr


S = 1/2                 # Spin
sample_step = 150
#b = 1
N = 20
k = 1

# Set up the calculations.
Sx, Sy, Sz = init(S)
Sx_k = get_full_matrix(Sx,k,N)
Sy_k = get_full_matrix(Sy,k,N)
Sz_k = get_full_matrix(Sz,k,N)
for delta in range(1,floor(N/2)+1):
    Sx_k_delta = get_full_matrix(Sx,k+delta,N)
    Sy_k_delta = get_full_matrix(Sy,k+delta,N)
    Sz_k_delta = get_full_matrix(Sz,k+delta,N)

    # Find S_k_total dot S_k_delta_total.
    S_product = [Sx_k*Sx_k_delta,Sy_k*Sy_k_delta,Sz_k*Sz_k_delta]
    S_product = np.sum(S_product)

    # Parallelize the plotting process
    b = np.linspace(-3,3,sample_step)
    pool = multiprocessing.Pool(processes=1)
    exp_values = pool.map(get_correlation,b)
    pool.close()

    plt.plot(b,exp_values)
plt.savefig('QM_3.17_correlations_20spins_delta_1-10.png')
