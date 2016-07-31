'''
This function generates a block diagonalized Hamiltonian for a
Heisenberg chain model.
'''

from quantum_module import rand_sign,permute_one_zero
import numpy as np
from scipy.sparse import lil_matrix
from scipy.misc import comb
import operator

def get_heisenberg_chain_H(Sx,Sy,Sz,N,h,J=1):
    global current_j_basis_set,block_size,i,k,current_j,j_max,tot,random_field
    
    random_field = h*np.random.rand(N)
    for i in range(N):
        random_field[i] *= rand_sign()
     
    D = Sx.get_shape()[0]**N
    H = lil_matrix((D,D),dtype=complex)
    j_max = int(round(0.5*N))
    current_j = j_max
    curr_pos = 0
    while current_j >= -1*j_max:
        # Side length of the current block.
        block_size = int(comb(N,int(j_max-current_j)))

        # Find all the binary representations of the current j.
        basis_set_seed = [0]*N
        for n in range(j_max-current_j):
            basis_set_seed[N-1-n] = 1
        current_j_basis_set = []
        if not (current_j == j_max or current_j == -1*j_max):
            for i in range(block_size):
                current_j_basis_set.append(permute_one_zero(basis_set_seed)[:])
        else:
            # The permute function cannot permute lists for which there is
            #  only one permutation.
            current_j_basis_set.append(basis_set_seed)
        
        # Fill in the diagonal elements for the current block.
        for i in range(block_size):
            h_sum = 0
            tot = 0
            for k in range(N):
                if current_j_basis_set[i][k] == 0:
                    h_sum -= random_field[k]
                elif current_j_basis_set[i][k] == 1:
                    h_sum += random_field[k]
                    
                diff = current_j_basis_set[i][N-1-k] 
                diff -= current_j_basis_set[i][N-2-k]
                tot += abs(diff)
            tot = N-2*tot
            H[curr_pos+i,curr_pos+i] = -1*tot*0.25 + 0.5*h_sum
            
        # Fill in the off-diagonal elements for the current block.
        if block_size != 1:
            for i in range(block_size):
                for j in range(block_size):
                    if i != j:
                        # All that says is that the only places that will 
                        # have a non-zero value is where there is one and only
                        # one spot between the bra and ket that 10-->01 or 
                        # 01-->10 happend. (Will fix comment later)
                        diff = list(map(operator.sub,current_j_basis_set[i],
                                  current_j_basis_set[j]))
                        diff = list(map(operator.abs,diff))
                        diff_sum = sum(diff)
                        if diff_sum == 2:
                            diff_k = 0
                            for k in range(N):
                                diff_k += abs(diff[k] - diff[k-1])
                            if diff_k == 2:
                                H[curr_pos+i,curr_pos+j] = -0.5
        curr_pos += block_size
        current_j -= 1
    return H
