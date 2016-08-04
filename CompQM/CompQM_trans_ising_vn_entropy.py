import numpy as np
from numpy.linalg import eigvalsh,eigh
from scipy.sparse import dok_matrix,eye,kron
import matplotlib.pyplot as plt
import multiprocessing

def init(S):
    '''
    Initialize calculations by computing Sx, Sy and Sz. All calculations
    are done using the z-basis.
    '''
    
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
    '''
    Build the S matrices in an N particle system.
    
    S is the operator/state we want to work on. The state must be
    put in a column vector form. S must be sparse.
    
    k is the location of the particle in a particle chain.
    '''
    
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
        Sx_sum += get_full_matrix(Sx,k,N)
    
        Sz_k = get_full_matrix(Sz,k,N)
        Sz_k_1 = get_full_matrix(Sz,k+1,N)
        
        Sz_sum += Sz_k*Sz_k_1
        
    H = -b/2 * Sx_sum - Sz_sum       # Hamiltonian
    
    return H

def get_red_rho_A(Sx,Sz,b,N):
    '''
    Form the reduced ground state density matrix.
    Only defined when N >= 2.
    '''
    
    if N < 2:
        raise Exception("N must be greater or equal to 2.")
    
    H = get_tran_ising_H(Sx,Sz,b,N)
    E,V = eigh(H.toarray())
    
    rho_A_B = np.outer(V[:,0],V[:,0])
    l,basis = eigh(Sz.toarray())
    rho_A_k = rho_A_B
    for k in range(N-1,0,-1):
        rho_A_m = np.zeros([D**k,D**k])
        for m in range(D):
            # Rewrite basis states in the full N space.
            basis_m = get_full_matrix(dok_matrix(basis)[m].transpose(),k,k)
            basis_m = np.kron(np.eye(D),basis_m.toarray())
            
            # Trace the density matrix over the k-th particle.
            rho_A_m += np.dot(np.dot(np.transpose(basis_m),rho_A_k),basis_m)
        rho_A_k = rho_A_m
    rho_A = rho_A_k
    
    return rho_A
    
def get_vn_entropy(b):
    '''Compute the von Neumann entropy for a given b.'''
    
    red_rho_A = get_red_rho_A(Sx,Sz,b,N)
    lamb = eigvalsh(red_rho_A)          # Eigenvalues of the reduced matrix.
    S_AB_terms = []
    for i in range(np.shape(red_rho_A)[0]):
        if abs(lamb[i]) < 1e-6:
            # lim a->0 (alog(a)) = 0. It also removes some minuscule negative 
            #  lambda values resulting from rounding errors.
            S_AB_terms.append(0)
        else:
            S_AB_terms.append(-lamb[i]*np.log2(lamb[i]))
        
    return np.sum(S_AB_terms)


S = 1/2                 # Spin
sample_step = 100
N = 3

Sx, Sy, Sz = init(S)
D = Sx.get_shape()[0]              # Dimensions of the spin matrices.

# Parallelize the plotting process
b = np.linspace(-3,3,sample_step)
pool = multiprocessing.Pool(processes=4)
entropy = pool.map(get_vn_entropy,b)
pool.close()

plt.plot(b,entropy)
plt.show()
