from quantum_module import init,show_elapsed_time,show_progress,get_vn_entropy
from mbl import get_random_state,get_init_delta_t,get_delta_delta_t
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply
from multiprocessing import Pool,Manager
import os
from time import time

def plot_entropy():
    entropy_plot = np.zeros(sample_size)
    init_delta_t,r = get_init_delta_t(time_range_lower_lim,
                        time_range_upper_lim,sample_size)
    H,E,psi = get_random_state(Sx,Sy,Sz,spin,N,h,mode='expm')
    
    # Plot the first point which does not require time evolution.
    entropy_plot[0] += get_vn_entropy(psi,spin,N,mode='eqsplit')

    # Plot the second point which requires the first time evolution.
    current_delta_t = init_delta_t
    psi_time_evolved = expm_multiply(-1j*H*current_delta_t,psi)
    entropy_plot[1] += get_vn_entropy(psi_time_evolved,
                                spin,N,mode='eqsplit')

    # Plot the rest of the points with time evolution.
    for plot_point in range(2,sample_size):
        delta_delta_t = get_delta_delta_t(time_range_lower_lim,plot_point,r)
        current_delta_t += delta_delta_t
        psi_time_evolved = expm_multiply(-1j*H*current_delta_t,
                                psi_time_evolved)
        entropy_plot[plot_point] += get_vn_entropy(psi_time_evolved,
                                        spin,N,mode='eqsplit')
    return entropy_plot

def get_plot(p):
    entropy_plot = plot_entropy()
    namespace.progress += 1
    show_progress(start_time,namespace.progress-1,psi_count)
    return entropy_plot

# Sets up the environment for computation.
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
init_delta_t,r = get_init_delta_t(time_range_lower_lim,
                    time_range_upper_lim,sample_size)
current_delta_t = init_delta_t
t.append(time_range_lower_lim+current_delta_t)
for plot_point in range(2,sample_size):
    delta_delta_t = get_delta_delta_t(time_range_lower_lim,plot_point,r)
    current_delta_t += delta_delta_t
    current_t += current_delta_t
    t.append(current_t)

# Find the average of many initial states.
manager = Manager()
namespace = manager.Namespace()
namespace.progress = 0
show_progress(start_time,namespace.progress,psi_count)
pool = Pool(processes=threads)
entropy_plot_list = pool.map(get_plot,range(psi_count))
pool.close()

# Average over all states.
entropy_plot = entropy_plot_list[0]
for n in range(1,psi_count):
    entropy_plot += entropy_plot_list[n]
entropy_plot /= psi_count

# Save the plot data.
datafilename = 'entropy_plot_data/entropy_plot_' + str(N) 
datafilename += 'particles_disorder' + str(h) + '_avg' + str(psi_count) 
datafilename += '_time' + str(time_range_upper_lim) + '.txt'
np.savetxt(datafilename,entropy_plot)
show_elapsed_time(start_time)

# Set up the plot.
plt.xlim(time_range_lower_lim,time_range_upper_lim*2)
plt.ylim(0.1,10)
plt.xlabel('Time (t)')
plt.ylabel('S(t)')
if h > 3.7:
    regime = 'MBL'
else:
    regime = 'ETH'
title = regime + ' entropy size=' + str(N) + ' h='
title += str(h) + ' averaged over ' + str(psi_count) + ' states'
plt.title(title)
plt.loglog(t,entropy_plot,linewidth=0, marker='o', 
        markerfacecolor='orangered', markeredgewidth=0, markersize=4.5)
figname = regime + '_entropy' + str(N) + 'particles_disorder' + str(h) 
figname += '_avg' + str(psi_count) + '.png'

if output_mode == str(1) or output_mode == str(3):
    plt.savefig(figname)
if output_mode == str(2) or output_mode == str(3):
    plt.show()
