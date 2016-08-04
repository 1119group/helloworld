from quantum_module import init, show_elapsed_time, show_progress, get_full_matrix
from mbl import get_random_state
import numpy as np
from scipy.sparse.linalg import expm
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt
from multiprocessing import Pool,Manager
import os
from time import time

def get_init_delta_t():
    log_scale_interval = np.log10(time_range_upper_lim/time_range_lower_lim)
    t_2 = time_range_lower_lim*10**(log_scale_interval/(sample_size-1))
    init_delta_t = t_2 - time_range_lower_lim
    r = t_2/time_range_lower_lim
    return init_delta_t,r
 
def get_delta_delta_t(plot_point,r):
    delta_delta_t = time_range_lower_lim*r**(plot_point-1)*(r-1)**2
    return delta_delta_t

def plot_imbalance(Sz,N,time_range,sample_size):
    init_delta_t,r = get_init_delta_t()
    H,E,psi = get_random_state(Sx,Sy,Sz,spin,N,h,mode='expm',seed=True)
    imbalance_plot = np.zeros(sample_size)
    psi_Szs = np.empty(N,dtype=object)
    full_Szs = np.empty(N,dtype=object)

    # Plot the first point which does not require time evolution.
    for k in range(N):
        Sz_full_k = get_full_matrix(Sz,k,N)
        psi_Sz = psi.transpose().conjugate().dot(Sz_full_k)
        full_Szs[k] = Sz_full_k.copy()
        psi_Szs[k] = psi_Sz.copy()
        imbalance_plot[0] += np.real(psi_Sz.dot(Sz_full_k.dot(psi))[0,0])

    # Plot the second point which requires the first time evolution.
    current_delta_t = init_delta_t
    U_delta_t = expm(-1j*H*current_delta_t)
    U_delta_t_dag = U_delta_t.transpose().conjugate()
    psi_time_evolved = U_delta_t.dot(psi)
    for k in range(N):
        psi_Szs[k] = psi_Szs[k].dot(U_delta_t_dag)
        imbalance_plot[1] += np.real(psi_Szs[k].dot(
                                        full_Szs[k].dot(
                                          psi_time_evolved))[0,0])

    # Plot the rest of the points with time evolution.
    for plot_point in range(2,sample_size):
        delta_delta_t = get_delta_delta_t(plot_point,r)
        current_delta_t += delta_delta_t
        U_delta_t = expm(-1j*H*current_delta_t)
        U_delta_t_dag = U_delta_t.transpose().conjugate()
        psi_time_evolved = U_delta_t.dot(psi_time_evolved)
        for k in range(N):
            psi_Szs[k] = psi_Szs[k].dot(U_delta_t_dag)
            imbalance_plot[plot_point] += np.real(psi_Szs[k].dot(
                                            full_Szs[k].dot(
                                              psi_time_evolved))[0,0])
    return 4/N*imbalance_plot

def get_plot(p):
    imbalance_plot = plot_imbalance(Sz,N,
                        time_range_upper_lim,sample_size)
    namespace.progress += 1
    show_progress(start_time,namespace.progress-1,psi_count)
    return imbalance_plot

# Set up the environment for computation.
start_time = time()
spin = 1/2
N = int(input('Enter the size "N" of the particle system: '))
h = int(input('Enter the disorder strength "h": '))
D = int(2*spin+1)**N
Sx,Sy,Sz = init(spin)
psi_count = int(input('Enter the number of states you are averaging over: '))
sample_size = int(input('Enter the number of points you are plotting: '))
threads = os.cpu_count()
time_range_lower_lim = 0.1
time_range_upper_lim = int(input('Enter the upper limit of "t": '))
output_mode_prompt = 'Exit behavior:\n\t1. Save the figure'
output_mode_prompt += '\n\t2. Display the figure on screen'
output_mode_prompt += '\n\t3. Save the figure and show it on screen'
output_mode_prompt += '\nPlease choose one option from the list above: '
output_mode = input(output_mode_prompt)
while True:
    if output_mode == str(1) or output_mode == str(2) or output_mode == str(3):
        break
    output_mode = input(
         'The only available options are 1, 2 and 3.\nPlease choose again: '
         )
print('')

# Find where to plot the points on a logarithmic time scale.
t = []
t.append(time_range_lower_lim)
current_t = time_range_lower_lim
init_delta_t,r = get_init_delta_t()
current_delta_t = init_delta_t
t.append(time_range_lower_lim+current_delta_t)
for plot_point in range(2,sample_size):
    delta_delta_t = get_delta_delta_t(plot_point,r)
    current_delta_t += delta_delta_t
    current_t += current_delta_t
    t.append(current_t)

# Find the average of many initial states.
manager = Manager()
namespace = manager.Namespace()
namespace.progress = 0
show_progress(start_time,namespace.progress,psi_count)
pool = Pool(processes=threads)
imbalance_plot_list = pool.map(get_plot,range(psi_count))
pool.close()

# Average over all states.
imbalance_plot = imbalance_plot_list[0]
for n in range(1,psi_count):
    imbalance_plot += imbalance_plot_list[n]
imbalance_plot /= psi_count

datafilename = 'imbalance_plot_data/imbalance_plot_' + str(N) 
datafilename += 'particles_disorder' + str(h) + '_avg' + str(psi_count) 
datafilename += '_time' + str(time_range_upper_lim) + '.txt'
np.savetxt(datafilename,imbalance_plot)
show_elapsed_time(start_time)

# Set up the plot.
plt.xlim(time_range_lower_lim,time_range_upper_lim*2)
plt.ylim(0,1)
plt.xlabel('Time (t)')
plt.ylabel('I(t)')
if h > 3.7:
    regime = 'MBL'
else:
    regime = 'ETH'
title = regime + ' imbalance size=' + str(N) + ' h='
title += str(h) + ' averaged over ' + str(psi_count) + ' states'
plt.title(title)

plt.semilogx(t,imbalance_plot,linewidth=0, marker='o', 
        markerfacecolor='orangered', markeredgewidth=0, markersize=4.5)

figname = regime + '_imbalance' + str(N) + 'particles_disorder' + str(h) 
figname += '_avg' + str(psi_count) + '.png'

if output_mode == str(1) or output_mode == str(3):
    plt.savefig(figname)
if output_mode == str(2) or output_mode == str(3):
    plt.show()
