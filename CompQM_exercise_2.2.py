from matplotlib.pyplot import imshow,show
from matplotlib import cm
from numpy import linspace,sin,meshgrid,zeros
from mpmath import pi

N = 100                     # "Resolution" of the image.
sum_max = 3                 # n_max of the summation.
x = linspace(0,1,N)
y = linspace(0,1,N)
xy = meshgrid(x, y, indexing='ij')
P = zeros([N,N],float)

for i in range(N):
    for j in range(N):
        for n in range(sum_max):
            # i/N and j/N equals x.
            P[i,j] += 2*sin(n*float(pi)*i/N)*sin(n*float(pi)*j/N)
         
imshow(P,origin="lower",extent=[0,1,0,1],cmap=cm.coolwarm)
show()