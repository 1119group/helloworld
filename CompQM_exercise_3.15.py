import numpy as np
from numpy.linalg import eigh

def init(S):
    '''Initialize calculations by computing Sx, Sy and Sz.'''
    
    # Create the raising and lowering operators.
    S_plus = np.zeros([int(2*S+1),int(2*S+1)],float)
    for i in np.arange(-S,S,1):
        m = -i-1
        S_plus[int(i+S),int(i+S+1)] = np.sqrt(S*(S+1)-m*(m+1))
    
    S_minus = np.transpose(S_plus)
    
    # Create Sx, Sy and Sz.
    Sx = 0.5*(S_plus + S_minus)
    Sy = -0.5j*(S_plus - S_minus)
    
    Sz = np.zeros([int(2*S+1),int(2*S+1)],float)
    for i in np.arange(S,-S-1,-1):
        Sz[-int(i-S),-int(i-S)] = i
        
    return Sx, Sy, Sz

def get_full_matrix(S,k,N):
    '''Build the S matrices in a N particle system.'''
    
    D = np.shape(S)[0]           # Dimensions of the spin matrices.

    if k == 0 or k == N:         # k = N is the periodic boundary condition.
        S_kp = np.copy(S)
        for i in range(N-1):
            S_kp = np.kron(S_kp,np.eye(D))
            
    elif k == 1:
        S_kp = np.eye(D)
        S_kp = np.kron(S_kp,S)
        for i in range(N-2):
            S_kp = np.kron(S_kp,np.eye(D))
            
    else:
        S_kp = np.eye(D)
        for i in range(k-2):
            S_kp = np.kron(S_kp,np.eye(D))
        
        S_kp = np.kron(S_kp,S)
        
        for i in range(N-k):
            S_kp = np.kron(S_kp,np.eye(D))
            
    return S_kp

def get_ising_H(Sx,Sy,Sz,b,n):
    '''Build the Hamiltonian for the N-particle system.'''
    
    Sx_sum = np.zeros([2**N,2**N])
    Sy_sum = np.zeros([2**N,2**N])
    Sz_sum = np.zeros([2**N,2**N])
    for k in range(N):
        Sz_sum += get_full_matrix(Sz,k,N)
    
        Sx_k = get_full_matrix(Sx,k,N)
        Sx_k_1 = get_full_matrix(Sx,k+1,N)
        Sy_k = get_full_matrix(Sy,k,N)
        Sy_k_1 = get_full_matrix(Sy,k+1,N)
        
        Sx_sum += np.dot(Sx_k,Sx_k_1)
        Sy_sum += np.dot(Sy_k,Sy_k_1)
        
    H = -b/2 * Sz_sum - (Sx_sum + Sy_sum)       # Hamiltonian
    
    return H
    

S = 1/2                 # Spin
b = 1                   # Coefficient in the Ising model Hamiltonian.
N = 12                  # Number of interacting particles.

Sx, Sy, Sz = init(S)
H = get_ising_H(Sx,Sy,Sz,b,N)

print(H)