import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# plt.hist(values, bins=num_bins)
# plt.title(stuffthatnamesit)
# plt.xlabel(stringsvalues)
# plt.ylabel("count")
# fig = plt.gcf()
# plt.hist(ent_list, bins=25, normed=1)
# plt.title("S/L Histogram L=10 40 h 25 psi 200 phi")
# plt.xlabel("S/L")
# plt.ylabel("count")
#
# plt.show()
#
# plt.hist(var_list, bins=25, normed=1)
# plt.title("Variance Histogram L=10 40 h 25 psi 200 phi")
# plt.xlabel("Variance")
# plt.ylabel("count")
#
# plt.show()
N = 12
h = 4.0
c = np.sqrt(2)-1
num_bins = 100
ent_hst_file = 'DATA/ent_hst_L' + str(N) + '_h' + str(h) + '_c' + str(round(c, 2)) + '.txt'
var_hst_file = 'DATA/var_hst_L' + str(N) + '_h' + str(h) + '_c' + str(round(c, 2)) + '.txt'

# rawr = np.loadtxt(ent_hst_file)
rawr = np.loadtxt(var_hst_file)
plt.hist(rawr, bins=num_bins)
plt.title('S/L Histogram N= ' + str(N) + ' h = ' + str(h) + ' c = ' + str(round(c,2)))
plt.xlabel("S/L")
plt.ylabel("Count")
plt.show()