# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:06:10 2019

"""

import numpy as np
import matplotlib.pyplot as plt


K_B = 1.38E-23                  # Boltzmann constant:           [m2 kg / (s2 K)]
TEMP = 300
Q = 1.602E-19
C = 3E8                         # Speed of light:               [m / s]
H = 6.626E-34                   # Plancks constant:             [m2 kg / s]
H_BAR = H / 2 / np.pi            
M_EFF = 0.16 * 9.109E-31        # Effective electron mass
WORK_FUNCTION = 0.475
ELECTRON_AFFINITY = 4.15
ELECTRON_AFFINITY = -0.5

N_V = 1E19 * 10 ** 6
N_D = 1.5E19 * 10 ** 6
EPSILON_0 = 1 / (4 * np.pi * 10 ** (-7) * C ** 2)
EPSILON_SI = 11.7

phi_b = ELECTRON_AFFINITY + 1.12 - WORK_FUNCTION
barrier_height = phi_b + K_B * TEMP / Q * np.log(N_D / N_V)
#barrier_height = 0.7
barrier_width = np.sqrt(2 * EPSILON_0 * EPSILON_SI * barrier_height / Q / N_D)


#p_tunneling = np.exp(-2*barrier_width/(H/2/np.pi)*np.sqrt((2*M*barrier_height)))
decaying_length = np.sqrt(2 * H_BAR ** 2 / (np.pi ** 2 * M_EFF * barrier_height * Q))
p_tunneling = np.exp( - barrier_width / decaying_length)


p_tunneling = np.exp( - 4 / 3 * barrier_width * np.sqrt((2 * M_EFF * phi_b * Q) / H_BAR ** 2))
j_tunneling = Q * N_D / 10 ** 6 * 10 ** 7 * p_tunneling

plt.plot(WORK_FUNCTION, 1000 * j_tunneling)
print(j_tunneling)
#plt.ylim((0,40))