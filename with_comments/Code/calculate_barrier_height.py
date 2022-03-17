
"""
Created on Mon May  6 13:44:07 2019
@author: Richard Faasse
"""

# #####################################################################################################################
# CALCULATE BARRIER HEIGHT
# #####################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

# #####################################################################################################################

K_B = 1.38E-23                           # Boltzmann constant:           [m2 kg / (s2 K)]
TEMP = 300								 # Temperature					 [K]
Q = 1.602E-19							 # Elementary charge:       	 [C / electron]
C = 3E8                                  # Speed of light:               [m /s]
H = 6.626E-34                            # Plancks constant:             [m2 kg / s]
H_BAR = H / 2 / np.pi            		 # Reduced Plancks constant:     [m2 kg / s]
M_EFF = 0.26 * 9.109E-31                 # Effective electron mass		 [Kg ]
										 # OLAYA ### Effective mass for conductivity calculations me = 0.26 * mo
WORK_FUNCTION = 4.975					 # Work function 				 [eV]
ELECTRON_AFFINITY = 4.8E15				 # Electron affinity 			 [units?]
										 ### OLAYA ### around 133.6 KJ/mol

# #####################################################################################################################

N_V = 2.8E15 * 10 ** 6

N_D = 5E19 * 10 ** 6

# #####################################################################################################################

EPSILON_0 = 1 / (4 * np.pi * 10 ** (-7) * C ** 2)

EPSILON_SI = 11.7

# #####################################################################################################################

phi_b = - ELECTRON_AFFINITY + WORK_FUNCTION

# #####################################################################################################################

barrier_height = phi_b + K_B * TEMP / Q * np.log(N_D / N_V)

#barrier_height = 0.7

barrier_width = np.sqrt(2 * EPSILON_0 * EPSILON_SI * barrier_height / Q / N_D)

# #####################################################################################################################

#decaying_length = np.sqrt(2*H_BAR**2/(np.pi**2*M_EFF*barrier_height*Q))
#p_tunneling = np.exp(-barrier_width/decaying_length)


p_tunneling = np.exp(- 4 / 3 * barrier_width * np.sqrt((2 * M_EFF * phi_b * Q) / H_BAR ** 2))
### OLAYA ###
j_tunneling = Q * N_D / 10 ** 6 * 10 ** 7 * np.exp(- 4 / 3 * barrier_width * np.sqrt((2 * M_EFF * phi_b * Q) / H_BAR ** 2))
### OLAYA ### 

# #####################################################################################################################

plt.plot(WORK_FUNCTION, j_tunneling, linewidth = 3)

plt.xlabel('Work function of the metal (eV)')

plt.ylabel('Maximum tunneling current (mA/cm2)')

plt.rcParams.update({'font.size': 14})

plt.title('N_D = 5E19')

plt.ylim((0,40))
 
plt.show()
 

# #####################################################################################################################

print(j_tunneling)

# #####################################################################################################################

