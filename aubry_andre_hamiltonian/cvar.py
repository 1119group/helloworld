import entropy as ent
import numpy as np
import quantum_module as qm


# N = 14
# spin = 1/2
# hmin = 1
# hmax = 5
# c = np.sqrt(2) - 1
# points = 17
# phis = 200
# num_psis = 120
# total_Sz = 0
# print("N=14")
# ent.ent_agr_var_plots(spin, N, hmin, hmax, points, c, num_psis, phis)
# # N = 12
# # num_psis = 60
# # print("N=12")
# # ent.ent_agr_var_plots(spin, N, hmin, hmax, points, c, num_psis, phis)
# # N = 10
# # num_psis = 30
# # print("N=10")
# # ent.ent_agr_var_plots(spin, N, hmin, hmax, points, c, num_psis, phis)
#
# N = 12
# spin = 1/2
# hmin = 1
# hmax = 5
# c = np.sqrt(2) - 1
# points = 17
# phis = 200
# num_psis = 70
# total_Sz = 0
# print("N=12")
# ent.ent_agr_var_plots(spin, N, hmin, hmax, points, c, num_psis, phis)
#
# N = 10
# spin = 1/2
# hmin = 1
# hmax = 10
# c = np.sqrt(2) - 1
# points = 19
# phis = 800
# num_psis = 25
# total_Sz = 0
# print("N=10")
# ent.ent_agr_var_plots(spin, N, hmin, hmax, points, c, num_psis, phis)

N = 10
spin = 1/2
hmin = 1
hmax = 5
c = np.sqrt(2) - 1
points = 21
phis = 1000
num_psis = 50
total_Sz = 0
print("N = ", N)
ent.ent_agr_var_plots(spin, N, hmin, hmax, points, c, num_psis, phis)