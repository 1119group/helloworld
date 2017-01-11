import quantum_module as qm
import block_diag as bd

N=12
Sx, Sy, Sz = qm.init(0.5)
print('check')
full_S = [[qm.get_full_matrix(S, k, N) for k in range(N)] for S in [Sx, Sy, Sz]]
U = bd.similarity_trans_matrix(N)