import quantum_module as qm
import block_diag as bd
import aubry_andre_block_twoleg_H as tH

N = 12
total_Sz = 0
Sx, Sy, Sz = qm.init(0.5)
to_ords = tH.create_complete_basis(N, total_Sz)[2]
full_S = [[qm.get_full_matrix(S, k, N) for k in range(N)] for S in [Sx, Sy, Sz]]
U = bd.similarity_trans_matrix(N)
