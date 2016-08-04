import quantum_module as qm
from aubry_andre_H import aubry_andre_H
from aubry_andre_block_H import aubry_andre_block_H
import numpy as np
from scipy.sparse import dok_matrix,lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.misc import comb
import sys
 
def get_state(Sx,Sy,Sz,N,h,c,phi=0,J=1):
    s = Sx.get_shape()[0]
    D = s**N
    H = aubry_andre_H(Sx,Sy,Sz,N,h,c,phi=0,J=1)

    E_max = eigsh(H,k=1,which='LA',maxiter=1e6,return_eigenvectors=False)
    E_min = eigsh(H,k=1,which='SA',maxiter=1e6,return_eigenvectors=False)
    E = np.append(E_min,E_max)

    # Create initial state.
    counter = 0
    # Create initial state psi with magnetization of 0. Here we first form 
    #  a binary number which has an equal number of 1's and 
    #  0's.
    index = np.zeros(N)
    for k in range(int(round(0.5*N))):
        index[k] = 1

    exit_status = False
    while True:
        counter += 1
        zero_Sz_basis_count = int(round(comb(N,0.5*N)))
        index = qm.permute_one_zero(index)
        # Then we convert the binary number into decimal and put a 1
        #  at the spot indicated by the binary into a zero vector. That
        #  represents a state with an equal number of up and down spins
        #  -- zero magnetization.
        s = 0
        for k in range(N):
            s += int(index[k] * 2**k)
        psi_0 = dok_matrix((D,1),complex)
        psi_0[s,0] = 1
        # Make sure psi's energy density is very close to 0.5.
        exp_val = psi_0.conjtransp().dot(H.dot(psi_0))
        e = (exp_val[0,0] - E[0]) / (E[-1] - E[0])
        if abs(e-0.5) < 0.001:
            break
        # Display an error message when no suitable state is found.
        elif counter > zero_Sz_basis_count:
           exit_status = True
           break
    return H, psi_0, exit_status

def get_state_blk(Sx,Sy,Sz,N,h,c,phi=0,J=1):
    s = Sx.get_shape()[0]
    D = s**N
    j = 0
    redo = True
    while redo:
        s = np.random.random_integers(1e9,size=(10,))
        H = aubry_andre_block_H(Sx,Sy,Sz,N,h,seed=s)
        
        E_max,eigvects_max = eigsh(H_norm,k=1,which='LA',maxiter=1e6)
        E_min,eigvects_min = eigsh(H_norm,k=1,which='SA',maxiter=1e6)
        E = np.append(E_min,E_max)
        
        # Create initial state.
        counter = 0
        while True:
            counter += 1
            # Generate a random state psi with total <Sz> = 0.
            center_block_size = int(comb(N,N/2))
            psi = lil_matrix((D,1),dtype=complex)
            rand_1 = np.random.randint(0,center_block_size)
            psi[int(0.5*(D-center_block_size))+rand_1,0] = 1

            # Make sure psi's energy density is very close to 0.5.
            exp_val = psi.transpose().dot(H_block.dot(psi))
            e = (np.real(exp_val[0,0]) - E[0]) / (E[-1] - E[0])
            if abs(e-0.5) < 0.001:
                redo = False
                j += 1
                break
            # Regenerate the Hamiltonian after failing to generate any state
            #  for a number of times.
            elif counter > 2*center_block_size:
                j += 1
                break
    return H_block,psi
