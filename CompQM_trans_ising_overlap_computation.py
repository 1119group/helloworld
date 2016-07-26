import numpy as np
from scipy.sparse import dok_matrix,eye,kron
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from multiprocessing import Process,Manager

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

def get_augmented_matrix(S,k,N):
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

def get_tran_ising_H(Sx,Sz,b,N):
    '''Build the Hamiltonian for the N-particle system.'''
    
    D = Sx.get_shape()[0]              # Dimensions of the spin matrices.

    Sx_sum = dok_matrix((D**N,D**N))
    Sz_sum = dok_matrix((D**N,D**N))

    for k in range(N):
        Sx_sum += get_augmented_matrix(Sx,k,N)
    
        Sz_k = get_augmented_matrix(Sz,k,N)
        Sz_k_1 = get_augmented_matrix(Sz,k+1,N)
        
        Sz_sum += Sz_k*Sz_k_1
        
    H = -b/2 * Sx_sum - Sz_sum       # Hamiltonian
    
    return H

def get_V(Sx,Sz,b,N):
    H = get_tran_ising_H(Sx,Sz,b,N)
    E,V = eigsh(H,k=1,which='LM')
    
    Vs.append(V)


S = 1/2                 # Spin
N = 8

Sx, Sy, Sz = init(S)

with Manager() as manager:
    Vs = manager.list()
    p1 = Process(target=get_V,args=(Sx,Sz,-100000000000,N))
    p2 = Process(target=get_V,args=(Sx,Sz,-100000000,N))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    product = np.dot(np.conj(np.transpose(Vs[0])),Vs[1])
    print(product**2)