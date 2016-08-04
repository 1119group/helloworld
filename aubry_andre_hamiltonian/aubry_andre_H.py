"""Sets up the Heisenberg chain model Hamiltonian."""

from quantum_module import get_full_matrix,rand_sign
import numpy as np
from scipy.sparse import dok_matrix

def aubry_andre_H(Sx,Sy,Sz,N,h,c,phi=0,J=1):
    """
    Sx, Sy, Sz are the x, y, and z spin matrices (in the z-basis). "N" is the 
    size of the particle chain. "h" is upper and lower limits of the static 
    random fields.
    The parameter "J" is optional. If not provided, it is defaulted to 1.
    """
    D = Sx.get_shape()[0]
    Sx_product_sum = dok_matrix((D**N,D**N))
    Sy_product_sum = dok_matrix((D**N,D**N))
    Sz_product_sum = dok_matrix((D**N,D**N))
    Sz_sum = dok_matrix((D**N,D**N))

    field = np.empty(N)
    for i in range(N):
        field[i] = h*np.cos(c*2*np.pi*i+phi)
        
    for k in range(N):
        Sz_sum += field[k]*get_full_matrix(Sz,k,N)
        
        if k == 0:
            Sx_k = get_full_matrix(Sx,k,N)
        else:
            Sx_k = Sx_k_1.copy()
        Sx_k_1 = get_full_matrix(Sx,k+1,N)
        Sx_product_sum += Sx_k.dot(Sx_k_1)
        del Sx_k
        
        if k == 0:
            Sy_k = get_full_matrix(Sy,k,N)
        else:
            Sy_k = Sy_k_1.copy()
        Sy_k_1 = get_full_matrix(Sy,k+1,N)
        Sy_product_sum += Sy_k.dot(Sy_k_1)
        del Sy_k
        
        if k == 0:
            Sz_k = get_full_matrix(Sz,k,N)
        else:
            Sz_k = Sz_k_1.copy()
        Sz_k_1 = get_full_matrix(Sz,k+1,N)
        Sz_product_sum += Sz_k.dot(Sz_k_1)
        del Sz_k
        
    H = Sz_sum - J*(Sx_product_sum + Sy_product_sum + Sz_product_sum)
    return H 
