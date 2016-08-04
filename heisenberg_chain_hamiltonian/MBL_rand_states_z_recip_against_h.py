from quantum_module import init,get_vn_entropy,sort_eigs
from heisenberg_chain_H import get_heisenberg_chain_H
import numpy as np
from numpy.linalg import eigh
from scipy.sparse import dok_matrix
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
from time import time

def get_entropy_diag(t):
    '''Using diagonalized Hamiltonian.'''
    psi = dok_matrix((D,1),dtype=complex)
    for i in range(np.shape(psi_0)[0]):
        psi += psi_0[i,0]*np.exp(-1j*E[i]*t)*eigvects[:,i]
    entropy = get_vn_entropy(Sx,Sy,Sz,psi,N,mode='eqslit')
    return entropy

def get_entropy_exp(t):
    '''Using exponentiated Hamiltonian.'''
    psi = expm(-1j*H*t).dot(psi_0)
    entropy = get_vn_entropy(Sx,Sy,Sz,psi,S,N,mode='eqsplit')
    return entropy
    
def get_z_1(t):
    entropy = get_entropy_exp(t)
    z_1 = np.log(entropy)/np.log(t)
    return z_1

S = 1/2
N = 12
#h = 10.0
D = int(2*S+1)**N
Sx,Sy,Sz = init(S)
epsilon = 0.001                 # Tolerance of energy density difference.
psi_count = 20
#sample_size = 100               # Graphing sample size.
threads = os.cpu_count()

z_avg_list = []
h = np.linspace(1,3.5,10)
print("Progress:\t Estimated time to completion:")
for k in range(10):
    # Find the average of many initial states.
    z_1_sum = np.zeros(psi_count)

    start_time = time()
    #H_seeds = list(range(2*psi_count))
    for i in range(psi_count):
        #j = 0
        redo = True
        while redo:
            H = get_heisenberg_chain_H(Sx,Sy,Sz,N,h[k])
            E_max,eigvects_max = eigsh(H,k=1,which='LA',maxiter=1e7)
            E_min,eigvects_min = eigsh(H,k=1,which='SA',maxiter=1e7)
            E = np.append(E_min,E_max)

            # Create initial state.
            counter = 0
            while True:
                counter += 1
                psi_0 = dok_matrix((D,1))
                rand_int = np.random.randint(0,D)
                psi_0[rand_int,0] = 1
                exp_val = psi_0.conjtransp().dot(H.dot(psi_0))
                e = (exp_val[0,0] - E[0]) / (E[-1] - E[0])
                if abs(e-0.5) < epsilon:
                    redo = False
                    #j += 1
                    break
                # Regenerate the Hamiltonian after failing to generate any states
                #  for a number of times.
                elif counter > D:
                    #j += 1
                    break

        #t = np.linspace(0,100,sample_size)
        #pool = Pool(processes=threads)
        #z_1 = pool.map(get_entropy_exp,t)
        #pool.close()
        #z_1_sum += z_1
        
        psi = expm(-1j*H*100).dot(psi_0)
        entropy = get_vn_entropy(Sx,Sy,Sz,psi,S,N,mode='eqsplit')
        z_1 = np.log(entropy)
        z_1_sum[i] = np.copy(z_1)
    z_1_avg = np.sum(z_1_sum)/psi_count
    z_avg_list.append(z_1_avg)
    # Calculate time used for progress report purposes.
    elapsed = time() - start_time
    ET_sec = elapsed/(k+1)*(10-k-1)
    Hr = int(ET_sec // 3600)
    Min = int((ET_sec % 3600) // 60)
    Sec = int(ET_sec % 60)
    ET = str(Hr) + ":" + str(Min) + ":" + str(Sec)
    # Show progress
    screen_output = str(k+1) + "/" + str(10) + "\t\t " + ET
    print(screen_output)
elapsed = time() - start_time
Hr = int(elapsed // 3600)
Min = int((elapsed % 3600) // 60)
Sec = int(elapsed % 60)
elapsed_time = str(Hr) + ":" + str(Min) + ":" + str(Sec)
print("Time elapsed: " + elapsed_time)
print("Done.")

plt.xlim((0,4))
plt.xlabel('h')
plt.ylabel('1/z')
plt.plot(h,z_avg_list,'go')
plt.show()
