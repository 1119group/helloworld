import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np

def get_density(x,y):
    D = 0
    for n in range(sum_max):
        # i/N and j/N equals x.
        D += 2*np.sin(n*np.pi*x)*np.sin(n*np.pi*y)
        
    return D


N = 70                     # "Resolution" of the image.
sum_max = 10               # n_max of the summation.
x = np.linspace(0,1,N)
y = np.linspace(0,1,N)
x, y = np.meshgrid(x,y)
z = get_density(x,y)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap=cm.coolwarm,
                       linewidth=0,antialiased=True)

fig.colorbar(surf, shrink=0.6, aspect=15)

plt.show()
