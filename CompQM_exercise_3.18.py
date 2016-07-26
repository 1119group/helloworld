import numpy as np
from numpy.linalg import eigh
from scipy.sparse import dok_matrix,eye,kron

def init(S):
    '''Initialize calculations by computing Sx, Sy and Sz.'''
    
    # Create the raising and lowering operators.
    S_plus = dok_matrix((int(2*S+1),int(2*S+1)),float)
    for i in np.arange(-S,S,1):
        m = -i-1
        S_plus[int(i+S),int(i+S+1)] = np.sqrt(S*(S+1)-m*(m+1))
    
    S_minus = S_plus.transpose()
    
    # Create Sx, Sy and Sz.
    Sx = 0.5*(S_plus + S_minus)
    Sy = -0.5j*(S_plus - S_minus)
    
    Sz = dok_matrix((int(2*S+1),int(2*S+1)),float)
    for i in np.arange(S,-S-1,-1):
        Sz[-int(i-S),-int(i-S)] = i
        
    return Sx, Sy, Sz

def get_full_matrix(S,k,N):
    '''Build the S matrices in an N particle system.'''
    
    D = S.get_shape()[0]         # Dimensions of the spin matrices.

    if k == 0 or k == N:         # k = N is the periodic boundary condition.
        S_kp = dok_matrix(S)
        for i in range(N-1):
            S_kp = kron(S_kp,eye(D))
            
    elif k == 1:
        S_kp = eye(D)
        S_kp = kron(S_kp,S)
        for i in range(N-2):
            S_kp = kron(S_kp,eye(D))
            
    else:
        S_kp = eye(D)
        for i in range(k-1):
            S_kp = kron(S_kp,eye(D))
        
        S_kp = kron(S_kp,S)
        
        for i in range(N-k-1):
            S_kp = kron(S_kp,eye(D))
            
    return S_kp

def get_Heisenberg_H(Sx,Sy,Sz,b,n):
    '''Build the Hamiltonian for the N-particle system.'''
    
    D = Sx.get_shape()[0]                   # Dimensions of the spin matrices.
    
    Sz_sum = dok_matrix((D**N,D**N))        # Contribution from the B field.
    Sx_int_sum = dok_matrix((D**N,D**N))    # Contribution from interactions.
    Sy_int_sum = dok_matrix((D**N,D**N))
    Sz_int_sum = dok_matrix((D**N,D**N))
    
    S_int_sum = [[Sx_int_sum,Sy_int_sum,Sz_int_sum],[Sx,Sy,Sz]]
    
    for k in range(N):
        # Compute the B-field contribution in the Hamiltonian.
        Sz_sum += get_full_matrix(Sz,k,N)
        
        # Compute the S-S interaction dot product in the Hamiltonian.
        for i in range(3):                  # For x, y and z orientations.
            i_k = get_full_matrix(S_int_sum[1][i],k,N)
            i_k_1 = get_full_matrix(S_int_sum[1][i],k+1,N)
            
            S_int_sum[0][i] += i_k*i_k_1
        S_dot_S_sum = Sx_int_sum + Sy_int_sum + Sz_int_sum
        
    # Interaction Hamiltonian.    
    H = -b/2 * Sz_sum - S_dot_S_sum
    
    return H
    

S = 1/2                 # Spin
b = 3                   # Coefficient in the Ising model Hamiltonian.
N = 3

Sx, Sy, Sz = init(S)
H = get_Heisenberg_H(Sx,Sy,Sz,b,N)

print(H.toarray())