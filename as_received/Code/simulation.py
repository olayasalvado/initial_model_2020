# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:44:07 2019

@author: Richard Faasse
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from physical_constants import Q, F, R, H, K_B, C
from device_parameters import VOLUME, CAPACITY, SOC, RHO, C_ELECTROLYTE, A, \
THICKNESS, AREA_ABSORBER
from model_settings import REAL_ABSORBER_DATA, REAL_SOLAR_DATA, \
TEMP_DEPENDENT_J0,  NUM_E_G, NUM_IV_POINTS, NUM_E_THERM, DYNAMIC_SOC, \
NUM_TIME_STEPS, charge_collection_method

from losses import J0_BV, J0_BV_ref, R_SERIES, E_A, ALPHA
from functions import calculate_flux_balance, calculate_iv_curve, IV_with_Auger
import datetime as dt
sys.path.insert(0,'/media/sf_python/initial_model_2020/as_received/input_data/')
input_data_path = '/media/sf_python/initial_model_2020/as_received/input_data/'


# =============================================================================
# Define the device parameters
# =============================================================================

if REAL_ABSORBER_DATA:
    NUM_E_G = 1
    E_G = np.array([1.12]) * Q
else:
    # Choose number of bandgap energies and the range or specific bandgap 
    # energy
    if NUM_E_G > 1:
        E_G = np.linspace(0.7, 2.5, NUM_E_G) * Q
    elif NUM_E_G == 1:
        E_G = np.array([1.12])*Q

# Choose the number of thermodynamic potentials and the range or specific
# thermodynamic potential
if NUM_E_THERM > 1:
    E_THERM = np.linspace(1.6, 0.2, NUM_E_THERM)
elif NUM_E_THERM == 1:
    E_THERM = np.array([0.83])

SOC = SOC*np.ones((NUM_E_G, NUM_E_THERM))


# =============================================================================
# Load the spectrum and temperature
# =============================================================================
if not isinstance(REAL_SOLAR_DATA, bool):
    sys.exit("Choose boolean value for REAL_SOLAR_DATA ")

elif REAL_SOLAR_DATA:
    DATE = '20180731'

    # Load real time solar spectra
    SPECTRAL_DATA = np.genfromtxt(input_data_path + DATE + '.csv', delimiter = ',')
    LABDA_SPECTRUM = np.arange(SPECTRAL_DATA[0, 4], SPECTRAL_DATA[0, 5] + 1, SPECTRAL_DATA[0, 6])
    POWER_SPECTRUM = SPECTRAL_DATA[:, 7:len(LABDA_SPECTRUM) + 7]


    DAYTIME = np.floor(SPECTRAL_DATA[:, 3] / 100) + np.remainder(SPECTRAL_DATA[:, 3], 100) / 60
    DT = DAYTIME[1] - DAYTIME[0]

    #Load temperature data
    OPEN_DATA = open(input_data_path + 'temperature_data_' + DATE + '.txt', 'r')
    COUNT = 0
    day_in_the_year = np.zeros((45000 * 12,))
    T_AIR = np.zeros((45000 * 12, 2))

    keep_temperature = np.ones((45000 * 12,), dtype = bool)

    LINE = 0
    
    for LINE in OPEN_DATA:
        # Extract temperature data
        stripped = LINE.strip()
        columns = stripped.split(',')
        T_AIR[COUNT, :] = columns[1:3]

        # Match temperature data with spectral data, by matching both date as
        # well as time
        date_string = columns[0]
        date_list = (date_string.split('/'))
        month = int(date_list[0])
        day = int(date_list[1])
        year = int(date_list[2])
        day_in_the_year[COUNT] = (dt.date(year, month, day) - dt.date(year, 1, 1)).days + 1
        times_in_day_spectral_data = SPECTRAL_DATA[SPECTRAL_DATA[:,2] == day_in_the_year[COUNT],3]
        keep_temperature[COUNT] = np.any(np.round(times_in_day_spectral_data) == (np.round(T_AIR[COUNT, 0] * 100)))

        COUNT = COUNT + 1

    T_AIR = T_AIR[keep_temperature,:]
    day_in_the_year = day_in_the_year[keep_temperature]
    day_in_the_year[-2:-1] = day_in_the_year[np.sum(day_in_the_year > 0) - 1]
    
    #delete empties
    day_in_the_year = day_in_the_year[day_in_the_year > 0]
    T_AIR = T_AIR[(T_AIR[:, 0] > 0)]
    T_AIR[:, 0] = np.round(T_AIR[:, 0] * 100)    
    T_AIR[:, 1] = T_AIR[:, 1] + 273.15
    NUM_TIME_STEPS = np.size(POWER_SPECTRUM, 0)

    del COUNT, columns, stripped, OPEN_DATA, LINE

elif not REAL_SOLAR_DATA:
    # Load AM15 spectrum

    SOLAR_SPECTRUM = np.transpose(np.load(input_data_path + 'solar_spectrum.npy'))
    POWER_SPECTRUM = np.zeros(np.shape(SOLAR_SPECTRUM[:, SOLAR_SPECTRUM[0, :] >= 290]))
    POWER_SPECTRUM[0, :] = SOLAR_SPECTRUM[1, SOLAR_SPECTRUM[0, :] >= 290]
    LABDA_SPECTRUM = SOLAR_SPECTRUM[0, SOLAR_SPECTRUM[0, :] >= 290]
    
    T_AIR = 300 * np.ones((NUM_TIME_STEPS, 2))
    DT = 1


# =============================================================================
# Load absorbance data
# =============================================================================

if REAL_ABSORBER_DATA:
    SI_DATA = np.load(input_data_path + 'SI_DATA.npy')
    B = SI_DATA[:, 5] * 1E-4 * 300
    ALPHA_SI = SI_DATA[:, 1]
    ALPHA_SI = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], ALPHA_SI)
    ABSORBANCE = 1 - np.exp(- ALPHA_SI * THICKNESS)

T_ELECTROLYTE = T_AIR[0, 1]    


# =============================================================================
# Initialize arrays
# =============================================================================

input_power = np.zeros((NUM_TIME_STEPS, ))
soc_correction = np.zeros((NUM_E_G, NUM_E_THERM))
current_limit = np.zeros(NUM_E_G)
iv_curve = np.zeros((2, NUM_E_G, NUM_IV_POINTS, NUM_TIME_STEPS))
sufficient_photovoltage = np.zeros((NUM_E_G, NUM_IV_POINTS, NUM_E_THERM))
operating_current = np.zeros((NUM_E_G, NUM_E_THERM))
efficiency = np.zeros((NUM_E_G, NUM_E_THERM, NUM_TIME_STEPS))
output_power = np.zeros((NUM_E_G, NUM_E_THERM, NUM_TIME_STEPS))
output_temperature = np.zeros((NUM_TIME_STEPS,))
output_temperature_electrolyte = np.zeros((NUM_TIME_STEPS,))
tmp = np.zeros((NUM_TIME_STEPS,))
current_plot = np.zeros((NUM_TIME_STEPS,))
ETA = np.linspace(0, 0.6, NUM_TIME_STEPS)
jlim_array = -np.linspace(0.1,0.01,NUM_TIME_STEPS)

START_TIME = time.time()


#%%
# =============================================================================
# Use one of the following parameter arrays if you want to vary them, make 
# sure NUM_TIME_STEPS > 1
# =============================================================================

#R_series_array = np.linspace(0,34.3,NUM_TIME_STEPS)
#SOC_array = np.linspace(0.05,0.95, NUM_TIME_STEPS)
#T_ARRAY = np.linspace(273.15, 373.15, NUM_TIME_STEPS)
#j0_bv_array = np.logspace(-1,-5, NUM_TIME_STEPS)
#alpha_array = np.linspace(0.25,0.75, NUM_TIME_STEPS)
#j0_bv_array = np.array((1000000000000, 4.45E-3, 1.46E-3, 2.1E-4, 5E-5 ))
#j0_bv_array[0] = 100000
#REFL_ARRAY = np.array((0, 0.05, 0.1, 0.2))

# =============================================================================


for kk in range(NUM_TIME_STEPS):
    
    # If you want a varying parameter, and uncommented one of the lines above,
    # uncomment the adequate line below:
    
#    SOC = SOC_array[kk]*np.ones((NUM_E_G, NUM_E_THERM))
#    J0_BV = j0_bv_array[kk]
#    R_SERIES = R_series_array[kk]
#    ALPHA = alpha_array[kk]
    
    if REAL_SOLAR_DATA:
        INCOMING_SPECTRUM = np.array((LABDA_SPECTRUM, POWER_SPECTRUM[kk, :]))
    else:
        INCOMING_SPECTRUM = np.array((LABDA_SPECTRUM, POWER_SPECTRUM[0, :]))

    INCOMING_SPECTRUM = np.transpose(INCOMING_SPECTRUM)

    for ii in range(NUM_E_G):
        if REAL_ABSORBER_DATA:
            current_limit[ii], temp_absorber, input_power[kk], q_in = \
            calculate_flux_balance(INCOMING_SPECTRUM, E_G[ii], T_AIR[kk, 1], ABSORBANCE, T_ELECTROLYTE, 0.6)   
#            temp_absorber = T_ARRAY[kk]

            ALPHA_SI = SI_DATA[:, 1] * (temp_absorber / 300) ** B
            ALPHA_SI = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], ALPHA_SI)
            N = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], SI_DATA[:, 2])
            K = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], SI_DATA[:, 3])
#            REFL = ((N-1) + K)**2/((N+1) + K)**2
            REFL = 0
            ABSORBANCE = (1 - np.exp(- ALPHA_SI * THICKNESS)) * (1 - REFL)
            ABSORBANCE[LABDA_SPECTRUM > 1440] = 0
            
            E_photon = H*C/(LABDA_SPECTRUM*1E-9)
            
            dE = np.zeros(np.shape(E_photon))
            dE[0:-1] = np.abs(E_photon[0:-1] - E_photon[1:])
            dE[-1] = dE[-2]
            J_photon = POWER_SPECTRUM[0, :] / E_photon * (1 - REFL)
            dlabda = np.zeros((len(LABDA_SPECTRUM),))
            dlabda[0:-1] = np.abs(LABDA_SPECTRUM[0:-1] - LABDA_SPECTRUM[1:])
            dlabda[-1] = dlabda[-2]
            
            v_oc = 0.7   #Initialize v_oc, to converge to it later

            # Iterate the temperature/v_oc calcululations a few times, to 
            # obtain convergence 
           
            for i in range(3):          
                current_limit[ii], temp_absorber, input_power[kk], q_in = \
                calculate_flux_balance(INCOMING_SPECTRUM, E_G[ii], T_AIR[kk, 1], ABSORBANCE, T_ELECTROLYTE, v_oc)

                b1 = (2 / H ** 3 * 1 ** 2 / C ** 2 * E_photon ** 2 * np.exp( - E_photon / K_B / temp_absorber))
                integral = np.sum(b1 * ABSORBANCE * dE)
                dark_saturation_current = Q * np.pi * integral / 10 ** 4
                    
                v_oc = (K_B * temp_absorber / Q * np.log(current_limit[ii] / dark_saturation_current))

                if charge_collection_method:
                    L_E = 350E-4
                    S = 80
                    THICKNESS = 350E-4
                    THICKNESS_N = 100E-7
                    THICKNESS_P = THICKNESS - THICKNESS_N
                    N_Z = 1000
                    z = np.linspace(0, THICKNESS, N_Z)
                    G = np.zeros(np.shape(z))
                    dz = z[1] - z[0]
                    for zz in range(N_Z):
                        G[zz] = np.sum(ALPHA_SI * J_photon * np.exp( - ALPHA_SI * z[zz]) * dlabda,0) / 10 ** 4
                    CP = 1 / (np.cosh((THICKNESS_N - z) / L_E) + np.sinh((THICKNESS_N - z) / L_E) * (np.sinh(z / L_E) 
                            + S * L_E * np.cosh(z / L_E)) / (np.cosh(z / L_E) + S * L_E * np.sinh(z / L_E)))
                    CP_2 = 1 / (np.cosh((THICKNESS_P - z) / L_E) + np.sinh((THICKNESS_P - z) / L_E) * (np.sinh(z / L_E) 
                              + S * L_E * np.cosh(z / L_E)) / (np.cosh(z / L_E) + S * L_E * np.sinh(z / L_E)))
                    CP_2 = np.flipud((CP_2))
                    CP[z > THICKNESS_N] = 0
                    CP_2[z < THICKNESS_N] = 0
                    CP_tot = CP + CP_2
                    current_limit[ii] = Q * np.sum(G[0:N_Z] * CP_tot[0:N_Z] * dz)
                
                # Implementing the Tiedje-Yablonivich method for dark
                # saturation current determination
                b1 = (2 / H ** 3 * 1 ** 2 / C ** 2 * E_photon ** 2 * np.exp( - E_photon / K_B / temp_absorber))
                integral = np.sum(b1 * ABSORBANCE * dE)
                dark_saturation_current = Q * np.pi * integral / 10 ** 4
                    
                v_oc = (K_B * temp_absorber / Q * np.log(current_limit[ii] / dark_saturation_current))

            

        else:
            ABSORBANCE = (LABDA_SPECTRUM * 1E-9 < H * C / E_G[ii])
            REFL = 0
            v_oc = E_G[ii]   #Initialize v_oc, to converge to it later
            E_photon = H * C / (LABDA_SPECTRUM * 1E-9)
            
            dE = np.zeros(np.shape(E_photon))
            dE[0:-1] = np.abs(E_photon[0:-1] - E_photon[1:])
            dE[-1] = dE[-2]
            J_photon = POWER_SPECTRUM[0, :] / E_photon * (1 - REFL)
            dlabda = np.zeros((len(LABDA_SPECTRUM),))
            dlabda[0:-1] = np.abs(LABDA_SPECTRUM[0:-1] - LABDA_SPECTRUM[1:])
            dlabda[-1] = dlabda[-2]            
            
            # Iterate a few time to converge to a solution
            for i in range(3):          
                current_limit[ii], temp_absorber, input_power[kk], q_in = \
                calculate_flux_balance(INCOMING_SPECTRUM, E_G[ii], T_AIR[kk, 1], ABSORBANCE, T_ELECTROLYTE, v_oc)

                dark_saturation_current = (Q * A * 2 * K_B * temp_absorber / H ** 3 / C ** 2 * (E_G[ii] ** 2 + 2 * K_B * temp_absorber * E_G[ii] 
                    + 2 * (K_B * temp_absorber) ** 2) * np.exp(- E_G[ii] / K_B / temp_absorber) / 10 ** 4)
                    
                v_oc = (K_B * temp_absorber / Q * np.log(current_limit[ii] / dark_saturation_current))
                # Uncomment the next line if you want to fix the temperature
#                temp_absorber = T_ARRAY[kk]

        if TEMP_DEPENDENT_J0:
            J0_BV = J0_BV_ref * np.exp(- E_A / R / temp_absorber)
            
#        v_oc = 0.51

        iv_curve[:, ii, :, kk], overpotential = \
        calculate_iv_curve(current_limit[ii], v_oc, J0_BV, temp_absorber, dark_saturation_current, R_SERIES, ALPHA)

        soc_correction[ii, :] = (R * temp_absorber / F * np.log(SOC[ii, :] ** 2 / (1 - SOC[ii, :]) ** 2))
        
    soc_correction[SOC >= 0.999999999] = 1000
    
    for jj in range(NUM_E_THERM):
        condition = numpy.matlib.repmat(E_THERM[jj] + soc_correction[:, jj], NUM_IV_POINTS, 1)
        condition = np.transpose(condition)
        sufficient_photovoltage = iv_curve[0, :, :, kk] > condition
        operating_current[:, jj] = (np.max(iv_curve[1, :, :, kk] * sufficient_photovoltage, 1))
        efficiency[:, jj, kk] = (operating_current[:, jj] * (E_THERM[jj] + soc_correction[:, jj]) / input_power[kk] * 100)
        output_power[:, jj, kk] = (operating_current[:, jj] * (E_THERM[jj] + soc_correction[:, jj]) * 1000)
        if DYNAMIC_SOC:
            SOC[:, jj] = (SOC[:, jj] + AREA_ABSORBER * operating_current[:, jj] * DT / CAPACITY)
            SOC[SOC >= 1] = 0.999999999


    dT_electrolyte = q_in / (RHO * VOLUME * C_ELECTROLYTE)
    # Equilibrate the electrolyte temperature to the morning air tempearature
    # if the day changes
    if REAL_SOLAR_DATA:
        
        if int(day_in_the_year[kk+1] - day_in_the_year[kk]) == 0:    
            T_ELECTROLYTE = T_ELECTROLYTE + dT_electrolyte * DT
        
        elif int(day_in_the_year[kk+1] - day_in_the_year[kk]) == 1:
            T_ELECTROLYTE = T_AIR[kk+1, 1]
        
        else:
            print(int(day_in_the_year[kk+1] - day_in_the_year[kk]))
            T_ELECTROLYTE = T_AIR[kk+1, 1]

        
    

    output_temperature[kk] = temp_absorber
    output_temperature_electrolyte[kk] = T_ELECTROLYTE
    
    ### OLAYA ### Next line is commented as per Richard indications
    #tmp[kk] = overpotential[np.argmin(np.abs(iv_curve[1,:,:,kk] - 0.01))]

    print("\r {}".format(np.round((kk+1) / NUM_TIME_STEPS * 100)), end = "")    

del ii, jj, kk
ELAPSED_TIME = time.time() - START_TIME
print('\n The elapsed time = ', ELAPSED_TIME)
test = np.squeeze(efficiency)

#plt.plot(DAYTIME, test)
# %%
