import quantum_module as qm
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from entropy import plot_entropy_time_evo_log,plot_entropy_time_evo_lin
from imbalance import plot_imbalance_time_evo_lin,plot_imbalance_time_evo_log
from time import time

def get_plot():
    if program_mode == 'entropy':
        if plot_mode == '1' or plot_mode == '3':
            plot,exit_status = plot_entropy_time_evo_lin(spin,N,h,c,phi,
                                time_range_lower_lim,
                                time_range_upper_lim,sample_size)
        elif plot_mode == '2' or plot_mode == '4':
            plot,exit_status = plot_entropy_time_evo_log(spin,N,h,c,phi,
                                time_range_lower_lim,
                                time_range_upper_lim,sample_size)
    elif program_mode == 'imbalance':
        if plot_mode == '1' or plot_mode == '3':
            plot,exit_status = plot_imbalance_time_evo_lin(spin,N,h,c,phi,
                                time_range_lower_lim,
                                time_range_upper_lim,sample_size)
        elif plot_mode == '2' or plot_mode == '4':
            plot,exit_status = plot_imbalance_time_evo_log(spin,N,h,c,phi,
                                time_range_lower_lim,
                                time_range_upper_lim,sample_size)
    return plot,exit_status

# Sets up the environment for computation.
if len(sys.argv) != 2:
    sysexit_msg = 'Usage: main.py <mode>' 
    sysexit_msg += '\nAvailable modes: "entropy", "imbalance"'
    sys.exit(sysexit_msg)
if sys.argv[1] != 'entropy' and sys.argv[1] != 'imbalance':
    sys.exit('Available modes: "entropy", "imbalance"')
program_mode = sys.argv[1]
spin = 1/2
N = int(input('Enter the size "N" of the particle system: '))
h = float(input('Enter the field strength "h": '))
c = 0.5*(3-np.sqrt(5))
phi = float(input('Enter the phase shift "phi": '))
sample_size = int(input('Enter the number of points you are plotting: '))
time_range_lower_lim = 0.1
time_range_upper_lim = float(input('Enter the upper limit of "t": '))
plot_mode_prompt = '\nAxes format:\n\t1. Linear\n\t2. Semilogx'
plot_mode_prompt += '\n\t3. Semilogy\n\t4. Loglog'
plot_mode_prompt += '\nPlease choose one option from the list above: '
plot_mode = input(plot_mode_prompt)
while True:
    if plot_mode == str(1) or plot_mode == str(2): 
        break
    if plot_mode == str(3) or plot_mode == str(4): 
        break
    plot_mode = input(
         'The only available options are 1, 2, 3 and 4.\nPlease choose again: '
         )
output_mode_prompt = '\nExit behavior:\n\t1. Save the figure'
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
dump_mode = input('\nShould the plot data be dumped into a file? (Y/n) ')
dump_mode = dump_mode.lower()
while True:
    if dump_mode == 'y' or dump_mode == 'n' or dump_mode == '':
        break
    dump_mode = input(
        'Please choose "y" or "n". Empty responses default to "y."'
        )
if dump_mode == '':
    dump_mode = 'y'
print('')

start_time = time()
plot,exit_status = get_plot()
if exit_status:
    exit_msg = 'For this Hamiltonian, no state with <Sz>=0 '
    exit_msg += 'has an energy density close to 0.5.'
    exit_msg += '\nTry using a different phase shift "phi."'
    sys.exit(exit_msg)
 

# Set up the t axis.
if plot_mode == '1' or plot_mode == '3':
    t = np.linspace(time_range_lower_lim,time_range_upper_lim,sample_size)
elif plot_mode == '2' or plot_mode == '4':
    t = qm.get_log_t_axis(time_range_lower_lim,
            time_range_upper_lim,sample_size)

# Save the plot data.
if program_mode == 'entropy':
    datafilename = 'entropy_plot_data/entropy_plot_' + 'size' + str(N) 
    figname = 'pngs/entropy_'
elif program_mode == 'imbalance':
    datafilename = 'imbalance_plot_data/imbalance_plot_' + 'size' + str(N)
    figname = 'pngs/imbalance_'
datafilename += '_field_strength' + str(h) + '_phase_shift' + str(phi)
datafilename += '_c' + str(c)
if plot_mode == '1' or plot_mode == '3':
    datafilename += '_linear'
elif plot_mode == '2' or plot_mode == '4':
    datafilename += '_log'
datafilename += str(sample_size) + 'pts_time' + str(time_range_upper_lim) 
datafilename += '.txt'
if dump_mode == 'y':
    np.savetxt(datafilename,plot)

qm.show_elapsed_time(start_time)

# Set up the plot.
plt.xlabel('Time (t)')
if sys.argv[1] == 'entropy':
    plt.ylabel('S(t)')
elif sys.argv[1] == 'imbalance':
    plt.ylabel('I(t)')

title = program_mode.title() + ': size=' + str(N) + ' h=' + str(h)
title += ' phi=' + str(phi) + ' c=' + str(c)
plt.title(title)

if plot_mode == '1':
    plt.ylim(0,5)
    plt.xlim(0,time_range_upper_lim)
    plt.plot(t,plot,linewidth=0, marker='o', 
        markerfacecolor='orangered', markeredgewidth=0, markersize=4.5)
elif plot_mode == '2':
    plt.ylim(0,5)
    plt.xlim(time_range_lower_lim,time_range_upper_lim)
    plt.semilogx(t,plot,linewidth=0, marker='o', 
        markerfacecolor='orangered', markeredgewidth=0, markersize=4.5)
elif plot_mode == '3':
    plt.ylim(0.1,10)
    plt.xlim(0,time_range_upper_lim)
    plt.semilogy(t,plot,linewidth=0, marker='o', 
        markerfacecolor='orangered', markeredgewidth=0, markersize=4.5)
elif plot_mode == '4':
    plt.ylim(0.1,10)
    plt.xlim(time_range_lower_lim,time_range_upper_lim*2)
    plt.loglog(t,plot,linewidth=0, marker='o', 
        markerfacecolor='orangered', markeredgewidth=0, markersize=4.5)

figname += 'size' + str(N)
figname += '_field_strength' + str(h) + '_phase_shift' + str(phi)
figname += '_c' + str(c) + '_time' + str(time_range_upper_lim)
figname += '.png'

if output_mode == str(1) or output_mode == str(3):
    plt.savefig(figname)
if output_mode == str(2) or output_mode == str(3):
    plt.show()
