# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:42:09 2019

@author: Richard
"""

import numpy as np
import scipy.optimize as opt
import numpy.matlib
import matplotlib.colors
import sys
sys.path.insert(0,'/media/sf_python/initial_model_2020/as_received/input_data/')
input_data_path = '/media/sf_python/initial_model_2020/as_received/input_data/'

Q = 1.602E-19                   # Elementary charge :           [C/electron]
N_A = 6.022E23                  # Avogadro's number             [-/mol]
F = Q*N_A                       # Faraday constant:             [C/mol]
R = 8.314                       # Gas constant:                 [J/mol/K]
H = 6.626E-34                   # Plancks constant:             [m2 kg/s]
K_B = 1.38E-23                  # Boltzmann constant:           [m2 kg/s2/K]
C = 3E8                         # Speed of light:               [m/s]
SIGMA = 5.67E-8                 # Stefan Boltzmann constant:    [J/s/m2/K4]
THICKNESS = 350E-4
#THICKNESS_N = 100E-7
THICKNESS_N = 100E-4
THICKNESS_P = THICKNESS-THICKNESS_N


# =============================================================================
# Define the resolutions of the model inputs
# =============================================================================

N = 10
NUM_IV_POINTS = 1000
NUM_E_G = 100
NUM_E_THERM = 100
T = np.linspace(270,350,N)
N_B = 1E14


# =============================================================================
# Define the device parameters
# =============================================================================

#E_G = np.linspace(0.7, 2.5, NUM_E_G)*Q
E_G = np.array([1.12]) * Q
#E_THERM = np.linspace(1.6, 0, NUM_E_THERM)
E_THERM = np.array([0.4])
AREA_ABSORBER = 0
CAPACITY = 0.4 * 0.025 * N_A * Q
SOC = 0.5 * np.ones((NUM_E_G, NUM_E_THERM))
A = 2
H_AIR = 0
H_ELECTROLYTE = 0 * 1000
jlim = 0.04
I_G = jlim / Q * 10 ** 4
n = 4
L = 0.001
a = 100000
K = 600
#    SI_DATA = np.loadtxt('si_abs.txt')
SI_DATA = np.load(input_data_path + 'SI_DATA.npy')
#    SI_DATA[:, 0] = SI_DATA[:, 0]*1000
#    SI_DATA[-1, 0] = np.max(LABDA_SPECTRUM)
B = SI_DATA[:, 5] * 1E-4 * 300
V_oc = np.zeros(N,)
current_limit = np.zeros(N,)
I_0 = np.zeros(N,)

integral = np.zeros(N,)
C_AUGER = 3.88E-31
N_i = 1.45E10
V = np.zeros(1000,)

def IV_with_Auger(V, dark_saturation_current, Q, THICKNESS, C_AUGER, N_i, current_limit, j, K_B, T):
    x = V * Q / K_B / T
    y = dark_saturation_current * np.exp(x) + 0 * Q * THICKNESS * C_AUGER * N_i ** 3 * np.exp(3 * x / 2) - current_limit + j
    return y

#L_E_array = np.linspace(100E-4, 1000E-4, 100)
L_E_array = np.array((350E-4,))
S_array = np.linspace(5,80, 5)
#S_array = np.array(())
#
jlmax = np.zeros((100,101))
tmp2 = np.zeros((1000,5))
iv_curve = np.zeros((2, 1, 1000, N))
#z = np.matlib.repmat(np.linspace(0, 350E-4, 100), np.size(LABDA_SPECTRUM, 0), 1)
z = np.linspace(0, THICKNESS, 1000)
dz = z[1]-z[0]
G = np.zeros(np.shape(z))
tmp = np.zeros(np.shape(T))
for i in range(np.size(T,0)):
    ALPHA_SI = SI_DATA[:, 1] * (T[i] / 300) ** B
    ALPHA_SI = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], ALPHA_SI)
    N = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], SI_DATA[:, 2])
    K = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], SI_DATA[:, 3])
    REFL = ((N - 1) + K) ** 2 / ((N + 1) + K) ** 2
    #    plt.plot(LABDA_SPECTRUM, np.log10(ALPHA_SI))
    #    plt.xlim(0,1400)
    #    plt.ylim(-8,7)
    #    plt.show()
    ABS_SI = (1 - np.exp( - ALPHA_SI * THICKNESS))
#    plt.plot(LABDA_SPECTRUM, ABS_SI)
#    ABS_SI = ALPHA_SI/(ALPHA_SI + 1/(4*N**2*THICKNESS))
#    plt.plot(LABDA_SPECTRUM, ABS_SI)
#    ABS_SI = (LABDA_SPECTRUM*1E-9 < H*C/E_G)*(LABDA_SPECTRUM*1E-9 > 665E-9)
#    ABS_SI = (LABDA_SPECTRUM*1E-9 < H*C/E_G)

#    plt.plot(LABDA_SPECTRUM, ABS_SI)
    ABS_SI[LABDA_SPECTRUM > 1440] = 0
#ABS_SI = np.load('ABS_SI.npy')
    E_photon = H * C / (LABDA_SPECTRUM * 1E-9)

    dE = np.zeros(np.shape(E_photon))
    b1 = 2 / H ** 3 * 1 ** 2 / C ** 2 * E_photon ** 2 * np.exp( - E_photon / K_B / T[i])
    dE[0:-1] = np.abs(E_photon[0:-1] - E_photon[1:])
    dE[-1] = dE[-2]
    J_photon = POWER_SPECTRUM[0, :] / E_photon * (1 - REFL)

    #J_photon = 8*np.pi*n**2/C**2*nu**2*np.exp(-E_photon/K_B/T)
#    integral[i] = np.sum(ABS_SI*J_photon*0.25/10**4*dE/H)
    dlabda = np.zeros((len(LABDA_SPECTRUM),))
    dlabda[0:-1] = np.abs(LABDA_SPECTRUM[0:-1] - LABDA_SPECTRUM[1:])
    dlabda[-1] = dlabda[-2]
    current_limit[i] = np.sum(J_photon * ABS_SI * dlabda) * Q / 10 ** 4
    V = np.linspace(0, 1, 1000)
    for kk in range(1000):
        G[kk] = np.sum(ALPHA_SI * J_photon * np.exp(- ALPHA_SI * z[kk]) * dlabda,0) / 10 ** 4
    
    for aa in range(1):
        L_E = L_E_array[aa]
        for bb in range(5):
            S = S_array[bb]
            CP = 1 / (np.cosh((THICKNESS_N - z) / L_E) + np.sinh((THICKNESS_N - z) / L_E) * (np.sinh(z / L_E) + S * L_E * np.cosh(z / L_E)) / (np.cosh(z / L_E) + S * L_E * np.sinh(z / L_E)))
            CP_2 = 1 / (np.cosh((THICKNESS_P - z) / L_E) + np.sinh((THICKNESS_P - z) / L_E) * (np.sinh(z / L_E) + S * L_E * np.cosh(z / L_E)) / (np.cosh(z / L_E) + S * L_E * np.sinh(z / L_E)))
            CP_2 = np.flipud((CP_2))
            CP[z > THICKNESS_N] = 0
            CP_2[z < THICKNESS_N] = 0
            CP_tot = CP + CP_2
            tmp2[:,bb] = CP_tot
            jlmax[aa,bb] = Q * np.sum(G[0:1000] * CP_tot[0:1000] * dz)
#            jlmax2 = Q*np.sum(G*CP_2*dz)
    
    integral[i] = np.sum(b1 * ABS_SI * dE)
    I_0[i] = Q * np.pi * integral[i] / 10 ** 4
#    I_0[i] = (Q*2*2*K_B*T[i]/H**3/C**2
#           * (E_G**2 + 2*K_B*T[i]*E_G
#           + 2*(K_B*temp_absorber)**2)
#           * np.exp(-E_G/K_B/T[i])
#           / 10**4)
#    I_0[i] = 1E-9
#    V_oc[i] = (K_B * T[i]/Q * np.log(current_limit[i]/ I_0[i]))
    x = V * Q / K_B / T[i]
    j = current_limit[i] - I_0[i] * np.exp(x) - Q * THICKNESS * C_AUGER * N_i ** 3 * np.exp( 3 * x / 2) 
#    j = np.max(jlmax) - I_0[i]*np.exp(x) - Q*THICKNESS*C_AUGER*N_i**3*np.exp(3*x/2) 

    iv_curve[:,0,:,i] = np.squeeze(np.array([[V], [j]]))
    plt.plot(V,j)
    plt.ylim(0, 1.1 * current_limit[i])
    V_oc = K_B * T[i] / Q * np.log(N_B * np.max(jlmax) / S / N_i ** 2 / Q)
    eff = j * V / 0.09
    plt.plot(V, eff)
    plt.ylim(0,0.5)
    tmp[i] = np.max(eff)
    V_oc = K_B * T[i] / Q * np.log(S ** 2 * np.max(jlmax) / S / 2 / N_i / Q)

plt.show()


#plt.plot(T,integral)
plt.plot(z * 10 ** 4,np.log(G) / np.max(np.log(G)), label = 'charge generation')
plt.plot(z * 10 ** 4,CP_tot, label = 'collection probability')
plt.show()
#plt.plot(z*10**4,G/np.max(G))
#
#plt.plot(z*10**4, CP_2)
#plt.ylim(0,1)

#print(current_limit)
#print(I0)
#plt.show()
    
#cmap = plt.cm.viridis
#cmaplist = [cmap(i) for i in range(cmap.N)]
#cmaplist[0:50] = []
#cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)
#plt.subplots(figsize=(10, 6))
#CF = plt.pcolormesh(L_E_array*10**4, S_array, np.transpose(jlmax*1000), cmap = cmap)
#CS = plt.contour(L_E_array*10**4, S_array, np.transpose(jlmax*1000), colors = 'k')
#plt.clabel(CS, fmt = '%1.1f', colors = 'k')
#plt.colorbar(CF)
#plt.xlabel('Diffusion length ($\mu m$)')
#plt.ylabel('S (cm$^{-1}$)')
#plt.rcParams.update({'font.size': 20})
#plt.savefig("tmp.png", format="png", dpi = 300)
plt.plot(T,tmp)
plt.plot(T, 0.12 * (1 - 0.0045 * (T - 298)))
plt.show()
        
#print(jlmax)
#print(jlmax2)

