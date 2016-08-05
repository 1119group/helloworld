import quantum_module as qm
from aubry_andre_common import get_state
import numpy as np
from scipy.sparse.linalg import expm

def plot_imbalance_time_evo_log(spin,N,h,c,phi,time_range_lower_lim,
                        time_range_upper_lim,sample_size):
    D = int(2*spin+1)**N
    Sx,Sy,Sz = qm.init(spin)
    init_delta_t,r = qm.get_init_delta_t(time_range_lower_lim,
                        time_range_upper_lim,sample_size)
    H,psi,exit_status = get_state(Sx,Sy,Sz,N,h,c,phi,J=1)
    if not exit_status:
        imbalance_plot = np.zeros(sample_size)
        psi_Szs = np.empty(N,dtype=object)
        full_Szs = np.empty(N,dtype=object)

        # Plot the first point which does not require time evolution.
        for k in range(N):
            Sz_full_k = qm.get_full_matrix(Sz,k,N)
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
            delta_delta_t = qm.get_delta_delta_t(time_range_lower_lim,plot_point,r)
            current_delta_t += delta_delta_t
            U_delta_t = expm(-1j*H*current_delta_t)
            U_delta_t_dag = U_delta_t.transpose().conjugate()
            psi_time_evolved = U_delta_t.dot(psi_time_evolved)
            for k in range(N):
                psi_Szs[k] = psi_Szs[k].dot(U_delta_t_dag)
                imbalance_plot[plot_point] += np.real(psi_Szs[k].dot(
                                                full_Szs[k].dot(
                                                  psi_time_evolved))[0,0])
    return 4/N*imbalance_plot,exit_status

def plot_imbalance_time_evo_lin(spin,N,h,c,phi,time_range_lower_lim,
                            time_range_upper_lim,sample_size):
    D = int(2*spin+1)**N
    Sx,Sy,Sz = qm.init(spin)
    imbalance_plot = np.zeros(sample_size)
    delta_t = (time_range_upper_lim-time_range_lower_lim)/(sample_size-1)
    H,psi,exit_status = get_state(Sx,Sy,Sz,N,h,c,phi,J=1)

    if not exit_status:
        U_delta_t = expm(-1j*H*delta_t)
        U_delta_t_dag = U_delta_t.transpose().conjugate()
        psi_Szs = np.empty(N,dtype=object)
        full_Szs = np.empty(N,dtype=object)

        # Plot the first point which does not require time evolution.
        for k in range(N):
            Sz_full_k = qm.get_full_matrix(Sz,k,N)
            psi_Sz = psi.transpose().conjugate().dot(Sz_full_k)
            psi_Szs[k] = psi_Sz.copy()
            full_Szs[k] = Sz_full_k.copy()
            imbalance_plot[0] += np.real(psi_Sz.dot(Sz_full_k.dot(psi))[0,0])

        # Plot the rest of the points.
        for plot_point in range(1,sample_size):
           psi = U_delta_t_dag.dot(psi)
           for k in range(N):
               psi_Szs[k] = psi_Szs[k].dot(U_delta_t)
               imbalance_plot[plot_point] += np.real(psi_Szs[k].dot(
                                                full_Szs[k].dot(
                                                    psi))[0,0])
    return 4/N*imbalance_plot,exit_status
