from quantum_module import init,sort_eigs
from tran_ising_H import get_tran_ising_H
import numpy as np
from scipy.sparse import dok_matrix,eye,kron
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from multiprocessing import Pool

def get_V(b):
    H = get_tran_ising_H(Sx,Sz,b,N)
    E,V = eigsh(H,k=1,which='LM')
    E,V = sort_eigs(E,V)
    return V
    
def get_overlap_inf(b):
    '''Plot the overlap of wavefunctions.'''
    H = get_tran_ising_H(Sx,Sz,b,N)
    E,V = eigsh(H)
    E,V = sort_eigs(E,V)
    product = np.dot(np.conj(np.transpose(Vs[1])),V[:,0])
    return product**2
    
def get_overlap_ninf(b):
    '''Plot the overlap of wavefunctions.'''
    H = get_tran_ising_H(Sx,Sz,b,N)
    E,V = eigsh(H)
    E,V = sort_eigs(E,V)
    product = np.dot(np.conj(np.transpose(Vs[0])),V[:,0])
    return product**2

def get_degen_grnd_V():
    H = get_tran_ising_H(Sx,Sz,0,N)
    E,V = eigsh(H,k=2,which='LM')
    E,V = sort_eigs(E,V)
    return V

def get_overlap_plus_degen_grnd(b):
    '''Plot the overlap of wavefunctions.'''
    H = get_tran_ising_H(Sx,Sz,b,N)
    E,V = eigsh(H)
    E,V = sort_eigs(E,V)
    product = np.dot(np.conj(np.transpose(V[:,0])),Plus_grnd_V)
    return product**2    
    
def get_overlap_minus_degen_grnd(b):
    '''Plot the overlap of wavefunctions.'''
    H = get_tran_ising_H(Sx,Sz,b,N)
    E,V = eigsh(H)
    E,V = sort_eigs(E,V)
    product = np.dot(np.conj(np.transpose(V[:,0])),Minus_grnd_V)
    return product**2        
    

S = 1/2                 # Spin
N = 9
sample_size = 60

Sx, Sy, Sz = init(S)

# Set up program for multithreaded plotting.
b = np.linspace(-3,3,sample_size)
b_inf = [-1000000,1000000]

# Compute wavefunctions for b approaching inf and -inf.
print("Compute wavefunctions for b approaching inf and -inf.")
pool = Pool(processes=2)
Vs = pool.map(get_V,b_inf)
pool.close

# Plot the overlap of wavefunctions for b = +inf.
print("Plot the overlap of wavefunctions for b = +inf.")
pool = Pool(processes=4)
Pinf = pool.map(get_overlap_inf,b)
pool.close()

plt.plot(b,Pinf)

# Plot the overlap of wavefunctions for b = -inf.
print("Plot the overlap of wavefunctions for b = -inf.")
pool = Pool(processes=4)
Pninf = pool.map(get_overlap_ninf,b)
pool.close()

plt.plot(b,Pninf)

# Plot the overlap of degenerate ground state wavefunctions.
print("Plot the overlap of degenerate ground state wavefunctions.")
b = np.linspace(-3,3,200)
Vs_degen = get_degen_grnd_V()
Plus_grnd_V = (Vs_degen[:,0] + Vs_degen[:,1]) / np.sqrt(2)
Minus_grnd_V = (Vs_degen[:,0] - Vs_degen[:,1]) / np.sqrt(2)

print("10%")
pool = Pool(processes=4)
P_plus_grnd = pool.map(get_overlap_plus_degen_grnd,b)
pool.close()
plt.plot(b,P_plus_grnd)

print("55%")
pool = Pool(processes=4)
P_plus_grnd = pool.map(get_overlap_minus_degen_grnd,b)
pool.close()
plt.plot(b,P_plus_grnd)

# Plot the overlap of ground up and down states.
#Vs_degen

plt.show()
