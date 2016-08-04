"""Creates the XY Ising model Hamiltonian."""

from quantum_module import get_full_matrix
from scipy.sparse import dok_matrix

def get_ising_XY_H(Sx,Sy,Sz,b,N):
    '''
    Build the Hamiltonian for the N-particle system using the XY Ising model.
    '''
    D = Sx.get_shape()[0]
    Sx_sum = dok_matrix((D**N,D**N))
    Sy_sum = dok_matrix((D**N,D**N))
    Sz_sum = dok_matrix((D**N,D**N))
    Sz_k_sum = dok_matrix((D**N,D**N))

    for k in range(N):
        Sz_sum += get_full_matrix(Sz,k,N)
        
        if k == 0:
            Sx_k = get_full_matrix(Sx,k,N)
        else:
            Sx_k = Sx_k_1.copy()
        Sx_k_1 = get_full_matrix(Sx,k+1,N)
        Sx_sum += Sx_k.dot(Sx_k_1)
        del Sx_k
        
        if k == 0:
            Sy_k = get_full_matrix(Sy,k,N)
        else:
            Sy_k = Sy_k_1.copy()
        Sy_k_1 = get_full_matrix(Sy,k+1,N)
        Sy_sum += Sy_k.dot(Sy_k_1)
        del Sy_k
    del Sx_k_1,Sy_k_1
        
    H = -b/2 * Sz_sum - (Sx_sum + Sy_sum)
    return H 
