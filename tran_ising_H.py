"""Creates the transvers Ising model Hamiltonian."""

from quantum_module import get_full_matrix
from scipy.sparse import dok_matrix

def get_tran_ising_H(Sx,Sz,b,N):
    '''
    Build the Hamiltonian for the N-particle system using the transverse
    Ising model.
    '''
    
    D = Sx.get_shape()[0]              # Dimensions of the spin matrices.

    Sx_sum = dok_matrix((D**N,D**N))
    Sz_sum = dok_matrix((D**N,D**N))

    for k in range(N):
        Sx_sum += get_full_matrix(Sx,k,N)
    
        Sz_k = get_full_matrix(Sz,k,N)
        Sz_k_1 = get_full_matrix(Sz,k+1,N)
        
        Sz_sum += Sz_k*Sz_k_1
        
    H = -b/2 * Sx_sum - Sz_sum       # Hamiltonian
    
    return H 
