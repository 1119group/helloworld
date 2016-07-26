'''
This provides the fermionic Aubry Andre model Hamiltonian with an equivalent
mathematical representation in terms of spins.
'''

from quantum_module import get_full_matrix
import numpy as np
from scipy.sparse import dok_matrix

def matrix_product_sum(Sn):
    '''Computes all the matrix products Snk * Snk+1'''
    for k in range(N):
        S_k = get_full_matrix(S,k,N)
        S_k_list.append(S_x_k)
    S_k_prod_sum = dok_matrix((D,D),dtype=complex)
    for k in range(N):
        if k == N-1:                        # Periodic boundary condition
            S_k_prod_sum += S_k_list[N-1].dot(S_k_list[0])
        else:
            S_k_prod_sum += S_k_list[k].dot(S_k_list[k+1])
    return S_k_prod_sum

def fermionic_aubry_andre_H(Sx,Sy,Sz,J,delta,beta,phi):
    '''
    "J" is the tunneling matrix element between neighboring sites.
    "delta" is the disorder strength.
    "beta" is the lattice periodicities.
    "phi" is the phase offset.
    '''
    D = int(2*spin+1)**N
    S_dot_sum = dok_matrix((D,D),dtype=complex)
    for Sn in [Sx,Sy,Sz]:
        S_dot_sum += matrix_product_sum(Sn)
    
    
    
