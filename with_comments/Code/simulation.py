
"""
Created on Mon May  6 13:44:07 2019
@author: Richard Faasse
"""

# #####################################################################################################################
# SIMULATION - MAIN FILE
# #####################################################################################################################

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from physical_constants import Q, F, R, H, K_B, C
from device_parameters import VOLUME, CAPACITY, SOC, RHO, C_ELECTROLYTE, A, THICKNESS, AREA_ABSORBER
from model_settings import REAL_ABSORBER_DATA, REAL_SOLAR_DATA, TEMP_DEPENDENT_J0, NUM_E_G, \
NUM_IV_POINTS, NUM_E_THERM, DYNAMIC_SOC, NUM_TIME_STEPS, charge_collection_method
from losses import J0_BV, J0_BV_ref, R_SERIES, E_A, ALPHA
from functions import calculate_flux_balance, calculate_iv_curve, IV_with_Auger
import datetime as dt
sys.path.insert(0,'/media/sf_python/initial_model_2020/with_comments/input_data/')
input_data_path = '/media/sf_python/initial_model_2020/with_comments/input_data/'

# #####################################################################################################################
# Define the device parameters
# #####################################################################################################################

if REAL_ABSORBER_DATA:

    NUM_E_G = 1
    ### OLAYA ### Just 1 band gap energy as real absorber data have been provided

    E_G = np.array([1.12]) * Q
    ### OLAYA ### 1.12 for Silicon bandgap
    ### OLAYA ### Eg(eV) ---> Eg(J) : 1 eV = 1.602E-19 J

else:

    # Choose number of bandgap energies and the range or specific bandgap energy
    
    if NUM_E_G > 1:

        E_G = np.linspace(0.7, 2.5, NUM_E_G) * Q
        ### OLAYA ### doi: 10.1039/C9SE00333A. Band gap range of of photoabsorber 0.7 eV - 2.5 eV
        ### OLAYA ### The band gap range is converted from eV to J

    elif NUM_E_G == 1:

        E_G = np.array([1.12]) * Q
        ### OLAYA ### 1.12 for Silicon bandgap
        ### OLAYA ### Eg(eV) ---> Eg(J) : 1 eV = 1.602E-19 J

# Choose the number of thermodynamic potentials and the range or specific thermodynamic potential

if NUM_E_THERM > 1:

    E_THERM = np.linspace(1.6, 0.2, NUM_E_THERM)
    ### OLAYA ### doi: 10.1039/C9SE00333A. Range of thermodymanic potentials 0.2 V - 1,6 V

elif NUM_E_THERM == 1:

    E_THERM = np.array([0.83])
    ### OLAYA ### Thesis File: 0.4 Vcell at 50SOC for (FE((CN)6)-4/Cu+2 is 0.4V
    ### OLAYA ### doi: 10.1039/C9SE00333A.
    ### OLAYA ### This value should be changed depending on the selected electrolytes
    ### OLAYA ### if this starts to change often, an user input should be created for this variable

SOC = SOC * np.ones((NUM_E_G, NUM_E_THERM))
### OLAYA ### Create a 1s matrix that will be filled later on

# #####################################################################################################################
# Load the spectrum and temperature
# #####################################################################################################################

if not isinstance(REAL_SOLAR_DATA, bool):
    sys.exit("Choose boolean value for REAL_SOLAR_DATA ")

elif REAL_SOLAR_DATA:

    DATE = '20180731'
    ### OLAYA ### Date value as per file name
    ### OLAYA ### if this starts to change often, an user input should be created for this variable
    
    # Load real time solar spectra

    SPECTRAL_DATA = np.genfromtxt(input_data_path + DATE + '.csv', delimiter = ',')
    ### OLAYA ### Load spectral data from text file
       
    LABDA_SPECTRUM = np.arange(SPECTRAL_DATA[0, 4], SPECTRAL_DATA[0, 5] + 1, SPECTRAL_DATA[0, 6])
    ### OLAYA ### wavelenght range of values
    ### OLAYA ### SPECTRAL_DATA[0,4] = 290
    ### OLAYA ### SPECTRAL_DATA[0,5] = 1650
    ### OLAYA ### SPECTRAL_DATA[0,6] = 1
    
    POWER_SPECTRUM = SPECTRAL_DATA[:, 7:len(LABDA_SPECTRUM) + 7]
    ### OLAYA ### Incoming power spectrum range of values
    ### OLAYA ### SPECTRAL_DATA[:, 7]: incoming power spectrum

    DAYTIME = np.floor(SPECTRAL_DATA[:, 3] / 100) + np.remainder(SPECTRAL_DATA[:, 3], 100) / 60
    ### OLAYA ### floor: largest integer i, i<=x. e.g floor(1.2) --> 1
    ### OLAYA ### Info example: 516 for 5:16 ---> to hours 5.267 hours
    ### OLAYA ### Left side: it calculate the hours
    ### OLAYA ### Rigth side: it converts the minutes left to hours
    
    DT = DAYTIME[1] - DAYTIME[0]
    ### OLAYA ### It provides the value of differential of time.

    # Load temperature data
    
    OPEN_DATA = open(input_data_path + 'temperature_data_' + DATE + '.txt', 'r')
    
    COUNT = 0

    day_in_the_year = np.zeros((45000 * 12,))
    ### OLAYA ### Assumption: (31 days = 44640 minutes) 45000 minutes maximum per month in 12 months: minutes per year 
   
    T_AIR = np.zeros((45000 * 12, 2))
    ### OLAYA ### Assumption: matrix with a T value for each minute in the year. with 2 columns
    ### OLAYA ### Create a matrix that will be filled with values later on

    keep_temperature = np.ones((45000 * 12,), dtype = bool)
    ### OLAYA ### Assumption: matrix with a T value for each minute in the year, with boolean results

    LINE = 0
    
    for LINE in OPEN_DATA:

        # Extract temperature data
                
        stripped = LINE.strip()
        ### OLAYA ### Takes the specific line and eliminates all the spaces between data

        columns = stripped.split(',')
        ### OLAYA ### data is separated, they split were ',' is, and they are kept as a list format
        
        T_AIR[COUNT, :] = columns[1:3]
        ### OLAYA ### columns[1]: time info (0.01, 0.02, 0.03, 0.04,...)
        ### OLAYA ### columns[2]: T info

        # Match temperature data with spectral data, by matching both date as well as time

        date_string = columns[0]
        ### OLAYA ### columns[0]: contains date information

        date_list = (date_string.split('/'))
        ### OLAYA ### data is separated, they split were '/' is, and they are kept as a list format

        month = int(date_list[0])
        ### OLAYA ### date_list[0] from columns[0]: month info 

        day = int(date_list[1])
        ### OLAYA ### date_list[1] from columns[0]: day info 

        year = int(date_list[2])
        ### OLAYA ### date_list[2] from columns[0]: year info

        day_in_the_year[COUNT] = (dt.date(year, month, day) - dt.date(year, 1, 1)).days + 1
        ### OLAYA ### asign a number per date that corresponds with the position of that date in the year

        times_in_day_spectral_data = SPECTRAL_DATA[SPECTRAL_DATA[:,2] == day_in_the_year[COUNT], 3]
        ### OLAYA ### SPECTRAL_DATA[:,2]: position of the specific date in the year
        ### OLAYA ### day_in_the_year[COUNT] is just one number of the date
        ### OLAYA ### If spectral_data is fully equal to day_in_the_year, selects full column 3 of spectral_data
        ### OLAYA ### it provides the same result as printing spectral_data[:,3]
        ### OLAYA ### SPECTRAL_DATA[:,3] contains time info. Example: 516 for 5:16 ---> to hours 5.267 hours

        keep_temperature[COUNT] = np.any ( np.round(times_in_day_spectral_data) == (np.round(T_AIR[COUNT, 0] * 100)) )
        ### OLAYA ### times_in_day_spectral_data: contains time info
        ### OLAYA ### T_AIR[COUNT,0]: contains time info
        ### OLAYA ### Only time values where date matched between T & spectral data, are kept on the matrix
        ### OLAYA ### When the loop is completed, this matrix is totally filled with True / False 

        COUNT = COUNT + 1

    T_AIR = T_AIR[keep_temperature,:]
    ### OLAYA ### Re-assign values depending on True / False as value / 0
    ### OLAYA ### Only those values where date matched between T & spectral data, are kept on the matrix

    day_in_the_year = day_in_the_year[keep_temperature]
    ### OLAYA ### Re-assign values depending on True / False as value / 0
    ### OLAYA ### Only those values where date matched between T & spectral data, are kept on the matrix

    day_in_the_year[-2:-1] = day_in_the_year[np.sum(day_in_the_year > 0) - 1]
    ### OLAYA ### What is really doing here ? 

    # delete empties

    day_in_the_year = day_in_the_year[day_in_the_year > 0]
    ### OLAYA ### Keep those that are over 0
    ### OLAYA ### Only those values where date matched between T & spectral data are kept on the matrix

    T_AIR = T_AIR[(T_AIR[:, 0] > 0)]
    ### OLAYA ### keep all values at first column that are over 0
    ### OLAYA ### Only those values where date matched between T & spectral data are kept on the matrix

    T_AIR[:, 0] = np.round(T_AIR[:, 0] * 100)    
    ### OLAYA ### Round all values from the first column 
    ### OLAYA ### * 100 is unit change for time

    T_AIR[:, 1] = T_AIR[:, 1] + 273.15
    ### OLAYA ### Unit change: change C degrees to Kelvin degres

    NUM_TIME_STEPS = np.size(POWER_SPECTRUM, 0)
    ### OLAYA ### NUM_TIME_STEPS used will be equal to the number of different power spectrum values

    del COUNT, columns, stripped, OPEN_DATA, LINE
    ### OLAYA ### delete COUNT, columns, stripped, OPEN_DATA, LINE

elif not REAL_SOLAR_DATA:

    # Load AM15 spectrum

    SOLAR_SPECTRUM = np.transpose(np.load(input_data_path + 'solar_spectrum.npy'))
    ### OLAYA ### transpose matrix of the loaded document solar_spectrum.npy

    POWER_SPECTRUM = np.zeros(np.shape(SOLAR_SPECTRUM[:, SOLAR_SPECTRUM[0, :] >= 290]))
    ### OLAYA ### SOLAR_SPECTRUM[0,:]: Wavelength values for the full solar spectrum (280 to 4000)
    ### OLAYA ### SPECTRAL_DATA[0,4] = 290, 
    ### OLAYA ### 290 lower wavelength value in range for absorption in this case

    POWER_SPECTRUM[0, :] = SOLAR_SPECTRUM[1, SOLAR_SPECTRUM[0, :] >= 290]
    ### OLAYA ### SOLAR_SPECTRUM[0,:]: Wavelength values for the full solar spectrum (280 to 4000)
    ### OLAYA ### Assumption: SOLAR_SPECTRUM[1,:]: spectral irradiance (W m-2)
    ### OLAYA ### Keeps the Spectral irradiance values of wavelengths over 290
    ### OLAYA ### It locates the solution values on column 0 instead of 1 (remains filled with 0s)

    LABDA_SPECTRUM = SOLAR_SPECTRUM[0, SOLAR_SPECTRUM[0, :] >= 290]
    ### OLAYA ### SOLAR_SPECTRUM[0,:]: Wavelength values for the full solar spectrum (280 to 4000)
    ### OLAYA ### Keeps the wavelengths values of wavelengths over 290

    T_AIR = 300 * np.ones((NUM_TIME_STEPS, 2))
    ### OLAYA ### Matrix will have 2 columns, for time and T
    ### OLAYA ### Why 300? Measured ? Assumed ? Optical properties os silicon taken at 300K. Check this out!

    DT = 1
    ### OLAYA ### It provides the differential of time.

# #####################################################################################################################
# Load absorbance data
# #####################################################################################################################

if REAL_ABSORBER_DATA:

    SI_DATA = np.load(input_data_path + 'SI_DATA.npy')
    ### OLAYA ### Load file SI.DATA.npy. Values of absorbance for Silicon Si
    ### OLAYA ### SI_DATA[:,0]: Wavelength (nm)
    ### OLAYA ### SI_DATA[:,1]: alpha, absorption coefficient (cm-1)
    ### OLAYA ### SI_DATA[:,2]: n = n + ki --> n, real component of refractive index
    ### OLAYA ### SI_DATA[:,3]: n = n + ki --> k, extinction coefficient
    ### OLAYA ### SI_DATA[:,4]: ? Check this out!
    ### OLAYA ### SI_DATA[:,5]: ? Check this out!

    B = SI_DATA[:, 5] * 1E-4 * 300
    ### OLAYA ### Assumption: 1E-4 * 300 is for unit change
    ### OLAYA ### SI_DATA[:,5]: ? Check this out!

    ALPHA_SI = SI_DATA[:, 1]
    ### OLAYA ### SI_DATA[:,1]: alpha, absorption coefficient (cm-1)

    ALPHA_SI = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], ALPHA_SI)
    ### OLAYA ### "one-dimensional linear interpolation for monotonically increasing sample points"
    ### OLAYA ### SI_DATA[:,0]: Wavelength (nm)
    ### OLAYA ### obtains alpha values for the wavelengths in LABDA_SPECTRUM from values on ALPHA_SI

    ABSORBANCE = 1 - np.exp(- ALPHA_SI * THICKNESS)
    ### OLAYA ### absorbance of silicon. 
    ### OLAYA ### THESIS: Equation 2.9

T_ELECTROLYTE = T_AIR[0, 1]    
### OLAYA ### T_ELECTROLITE will be 300 if not real_solar_data. Matrix filled with 300s. 
### OLAYA ### T_ELECTROLYTE will be T_AIR[0,1] if real_solar_data. Why this specific value? Check this out!

# #####################################################################################################################
# Initialize arrays
# #####################################################################################################################

input_power = np.zeros((NUM_TIME_STEPS, ))
### OLAYA ### matrix filled with 0s. row:1 & columns: NUM_TIME_STEPS

soc_correction = np.zeros((NUM_E_G, NUM_E_THERM))
### OLAYA ### matrix filled with 0s. row: NUM_E_G & columns: NUM_E_THERM

current_limit = np.zeros(NUM_E_G)
### OLAYA ### matrix filled with 0s. row:1 & columns: NUM_E_G

iv_curve = np.zeros((2, NUM_E_G, NUM_IV_POINTS, NUM_TIME_STEPS))
### OLAYA ### 2 groups of matrix, number of matrix in each group: NUM_E_G, 
### OLAYA ### with rows: NUM_IV_POINTS & columns: NUM_TIME_STEPS

sufficient_photovoltage = np.zeros((NUM_E_G, NUM_IV_POINTS, NUM_E_THERM))
### OLAYA ### number of matrixs: NUM_E_G, with rows: NUM_IV_POINTS & columns: NUM_E_THERM

operating_current = np.zeros((NUM_E_G, NUM_E_THERM))
### OLAYA ### matrix filled with 0s. rows: NUM_E_G & columns: NUM_E_THERM

efficiency = np.zeros((NUM_E_G, NUM_E_THERM, NUM_TIME_STEPS))
### OLAYA ### matrix filled with 0s. number of matrixs: NUM_E_G, with rows: NUM_E_THERM & columns: NUM_TIME_STEPS

output_power = np.zeros((NUM_E_G, NUM_E_THERM, NUM_TIME_STEPS))
### OLAYA ### matrix filled with 0s. number of matrixs: NUM_E_G, with rows: NUM_E_THERM & columns: NUM_TIME_STEPS

output_temperature = np.zeros((NUM_TIME_STEPS,))
### OLAYA ### matrix filled with 0s. row:1 & columns: NUM_TIME_STEPS

output_temperature_electrolyte = np.zeros((NUM_TIME_STEPS,))
### OLAYA ### matrix filled with 0s. row:1 & columns: NUM_TIME_STEPS

tmp = np.zeros((NUM_TIME_STEPS,))
### OLAYA ### matrix filled with 0s. row:1 & columns: NUM_TIME_STEPS
### OLAYA ### Create a matrix that will be filled with values later on

current_plot = np.zeros((NUM_TIME_STEPS,))
### OLAYA ### matrix filled with 0s. row:1 & columns: NUM_TIME_STEPS

ETA = np.linspace(0, 0.6, NUM_TIME_STEPS)
### OLAYA ### start at 0, finish at 0.6, steps: NUM_TIME_STEPS
### OLAYA ### Why these values? 0 & 0.6 ? Check this out!

jlim_array = - np.linspace(0.1,0.01,NUM_TIME_STEPS)
### OLAYA ### start: - 0.1, finish: - 0.01, steps: NUM_TIME_STEPS
### OLAYA ### Why these values? -0.1 & -0.01 ? Check this out!

START_TIME = time.time()

# #####################################################################################################################
# Use one of the following parameter arrays if you want to vary them, make sure NUM_TIME_STEPS > 1
# #####################################################################################################################

#R_series_array = np.linspace(0,34.3,NUM_TIME_STEPS)
### OLAYA ### return NUM_TIME_STEPS numbers evenly spaced between 0 and 34.3
### OLAYA ### Why these values? 0 and 34.3 ? Check this out!

#SOC_array = np.linspace(0.05,0.95, NUM_TIME_STEPS)
### OLAYA ### return NUM_TIME_STEPS numbers evenly spaced between 0.05 and 0.95
### OLAYA ### Why these values? 0.05 and 0.95 ? Check this out!

#T_ARRAY = np.linspace(273.15, 373.15, NUM_TIME_STEPS)
### OLAYA ### return NUM_TIME_STEPS numbers evenly spaced between 273.15 and 373.15

#j0_bv_array = np.logspace(-1,-5, NUM_TIME_STEPS)
### OLAYA ### return NUM_TIME_STEPS numbers evenly spaced in a log scale between -1 and -5
### OLAYA ### Why these values? -1 and -5 ? Check this out!

#alpha_array = np.linspace(0.25,0.75, NUM_TIME_STEPS)
### OLAYA ### return NUM_TIME_STEPS numbers evenly spaced between 0.25 and 0.75
### OLAYA ### Why these values? 0.25 and 0.75 ? Check this out!

#j0_bv_array = np.array((1000000000000, 4.45E-3, 1.46E-3, 2.1E-4, 5E-5 ))
### OLAYA ### creates a matrix with these values
### OLAYA ### Why these values? 1000000000000, 4.45E-3, 1.46E-3, 2.1E-4, 5E-5 ? Check this out!

#j0_bv_array[0] = 100000
### OLAYA ### modify a value
### OLAYA ### Why this value? 100000? Check this out!

#REFL_ARRAY = np.array((0, 0.05, 0.1, 0.2))
### OLAYA ### creates a matrix with these values
### OLAYA ### Why these values? 0, 0.05, 0.1, 0.2 ? Check this out!

# #####################################################################################################################

for kk in range(NUM_TIME_STEPS):
    
    # If you want a varying parameter, and uncommented one of the lines above, uncomment the adequate line below:
    
    # SOC = SOC_array[kk]*np.ones((NUM_E_G, NUM_E_THERM))

    # J0_BV = j0_bv_array[kk]

    # R_SERIES = R_series_array[kk]
 
    # ALPHA = alpha_array[kk]

    if REAL_SOLAR_DATA:

        INCOMING_SPECTRUM = np.array((LABDA_SPECTRUM, POWER_SPECTRUM[kk, :]))
        ### OLAYA ### Power spectrum if real solar data: incoming power spectrum range of values
        ### OLAYA ### array first row: LABDA_SPECTRUM & array second row: POWER_SPECTRUM[kk, :]
    
    else:

        INCOMING_SPECTRUM = np.array((LABDA_SPECTRUM, POWER_SPECTRUM[0, :]))
        ### OLAYA ### Power spectrum if not real solar data: incoming power spectrum over 290 nm
        ### OLAYA ### array first row: LABDA_SPECTRUM & array second row: POWER_SPECTRUM[0, :]

    INCOMING_SPECTRUM = np.transpose(INCOMING_SPECTRUM)
    ### OLAYA ### array first column: LABDA_SPECTRUM & array second column: POWER_SPECTRUM[kk/0, :]

    for ii in range(NUM_E_G):

        if REAL_ABSORBER_DATA:

            current_limit[ii], temp_absorber, input_power[kk], q_in = \
            calculate_flux_balance(INCOMING_SPECTRUM, E_G[ii], T_AIR[kk, 1], ABSORBANCE, T_ELECTROLYTE, 0.6)
            ### OLAYA ### INPUTS: INCOMING_SPECTRUM, E_G, T_AIR, ABSORBANCE, T_ELECTROLYTE, v_oc
            ### OLAYA ### OUTPUT: current_limit, temp_absorber, input_power, q_in

            # temp_absorber = T_ARRAY[kk]
            ### OLAYA ### Assign specific loop value
            ### OLAYA ### In which conditions should be commented / uncommented. Check this out!

            ALPHA_SI = SI_DATA[:, 1] * (temp_absorber / 300) ** B
            ### OLAYA ### SI_DATA[:,1]: alpha, absorption coefficient (cm-1)
            ### OLAYA ### B = SI_DATA[:, 5] * 1E-4 * 300
            ### OLAYA ### CHECK THIS OUT ! AND  ALSO ANY OTHER LINE WITH B

            ALPHA_SI = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], ALPHA_SI)
            ### OLAYA ### "one-dimensional linear interpolation for monotonically increasing sample points"
            ### OLAYA ### alpha: absorption coefficient (cm-1)
            ### OLAYA ### SI_DATA[:,0]: Wavelength (nm)
            ### OLAYA ### obtains alpha values for wavelengths in LABDA_SPECTRUM from values on ALPHA_SI

            N = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], SI_DATA[:, 2])
            ### OLAYA ### "one-dimensional linear interpolation for monotonically increasing sample points"
            ### OLAYA ### SI_DATA[:,0]: Wavelength (nm)
            ### OLAYA ### SI_DATA[:,2]: n = n + ki --> n, real component of refractive index
            ### OLAYA ### obtains n values for wavelengths in LABDA_SPECTRUM from values on SI_DATA[:, 2]

            K = np.interp(LABDA_SPECTRUM, SI_DATA[:, 0], SI_DATA[:, 3])
            ### OLAYA ### "one-dimensional linear interpolation for monotonically increasing sample points"
            ### OLAYA ### SI_DATA[:,0]: Wavelength (nm)
            ### OLAYA ### SI_DATA[:,3]: n = n + ki --> k, extinction coefficient
            ### OLAYA ### obtains k values for wavelengths in LABDA_SPECTRUM from values on SI_DATA[:, 3]

            # REFL = ( (N - 1) + K ) ** 2 / ( (N + 1) + K ) ** 2
            ### OLAYA ### Thesis equation 2.20
            ### OLAYA ### Thesis assumption: normal ligth incidence. The angle of incidence is always 0. 
            ### OLAYA ### Air, n2 = 1 
            
            REFL = 0
            ### OLAYA ### Considering that no reflection is happening

            ABSORBANCE = (1 - np.exp(- ALPHA_SI * THICKNESS)) * (1 - REFL)
            ### OLAYA ### 1 = ABSORBANCE + REFLECTANCE + TRANSMITANCE
            ### OLAYA ### Thesis equation 2.21: Total reflectance = (1-REFL)
            ### OLAYA ### Incoming spectrum for the semiconductor = Incoming spectrum * (1-REFL)
            ### OLAYA ### This final incoming spectrum is the one that should be integrated for the calculations       

            ABSORBANCE[LABDA_SPECTRUM > 1440] = 0
            ### OLAYA ### top wavelength limit for absorbance: 1440   
            
            E_photon = H * C / (LABDA_SPECTRUM * 1E-9)
            ### OLAYA ### Thesis section 2.1 equation 2.1
            ### OLAYA ### Wavelengths in LABDA_SPECTRUM are transformed from nm to m 
    
            dE = np.zeros(np.shape(E_photon))
            ### OLAYA ### matrix filled with 0s: as per the np.shape of E_photon

            dE[0:-1] = np.abs(E_photon[0:-1] - E_photon[1:])
            ### OLAYA ### provides a matrix with the differential values of labda, element by element

            dE[-1] = dE[-2]
            ### OLAYA ### last value takes the same value of value-before-last-value 

            J_photon = POWER_SPECTRUM[0, :] / E_photon * (1 - REFL)
            ### OLAYA ### in functions file: J_photon = INCOMING_SPECTRUM[:, 1] / E_photon
            ### OLAYA ### Power spectrum in [0,:] divided by Ephoton & multiplied for (1_REFL)
            ### OLAYA ### Power spectrum in [0,:] divided by (h c / labda_spectrum) & multiplied by (1-REFL)
            ### OLAYA ### Thesis equation 2.21: Total reflectance = (1-REFL)
            ### OLAYA ### Incoming spectrum for the semiconductor = Incoming spectrum * (1-REFL)
            ### OLAYA ### This final incoming spectrum is the one that should be integrated for the calculations

            dlabda = np.zeros((len(LABDA_SPECTRUM),))
            ### OLAYA ### Matrix row: 1 & columns: len of labda_spectrum
       
            dlabda[0:-1] = np.abs(LABDA_SPECTRUM[0:-1] - LABDA_SPECTRUM[1:])
            ### OLAYA ### provides a matrix with the differential values of labda, element by element

            dlabda[-1] = dlabda[-2]
            ### OLAYA ### last value takes the same value of value-before-last-value 

            # Initialize v_oc, to converge to it later
            v_oc = 0.7   

            # Iterate the temperature / v_oc calcululations a few times, to obtain convergence 
           
            for i in range(3):   

                current_limit[ii], temp_absorber, input_power[kk], q_in = \
                calculate_flux_balance(INCOMING_SPECTRUM, E_G[ii], T_AIR[kk, 1], ABSORBANCE, T_ELECTROLYTE, v_oc)
                ### OLAYA ### INPUTS: INCOMING_SPECTRUM, E_G, T_AIR, ABSORBANCE, T_ELECTROLYTE, v_oc
                ### OLAYA ### OUTPUT: current_limit, temp_absorber, input_power, q_in

                ### OLAYA ### Implementing the Tiedje-Yablonivich method for dark saturation current determination
                ### OLAYA ### LIMITING EFFICIENCY OF SILICON SOLAR CELLS, 
                ### OLAYA ### IEEE TRANSACTIONS ON ELECTRON DEVICES, VOL. ED-31, NO.5, MAY 1984
  
                b1 = (2 / H ** 3 * 1 ** 2 / C ** 2 * E_photon ** 2 * np.exp( - E_photon / K_B / temp_absorber))
                ### OLAYA ### Equation 2 using n=1 from LIMITING EFFICIENCY OF SILICON SOLAR CELLS
                ### OLAYA ### n: index of refraction of the medium. n(air) = 1
                ### OLAYA ### The 1 in the denominator of Bose-Einstein thermal occupation factor has been neglected
                ### OLAYA ### Valid in the limit Eg - u > Kb * T
                ### OLAYA ### u is the internal chemical potential, constant throughout the material
                ### OLAYA ### u equals to the separation between the electron and hole quasi-Fermi levels 
                ### OLAYA ### u equals to solar cell output voltage if ideal contacts and no internal conc. gradient

                integral = np.sum(b1 * ABSORBANCE * dE)
                ### OLAYA ### To simplify the use of the following equation

                dark_saturation_current = Q * np.pi * integral / 10 ** 4
                ### OLAYA ### Equation 5 from LIMITING EFFICIENCY OF SILICON SOLAR CELLS
                ### OLAYA ### Perfect reflection is assumed: radiative coupling through back surface is neglected
                ### OLAYA ### E-4 for a unit change
                ### OLAYA ### Check this out! Thesis equation 2.4 same as this with solved integral 

                v_oc = (K_B * temp_absorber / Q * np.log(current_limit[ii] / dark_saturation_current))
                ### OLAYA ### Thesis equation A.4
                ### OLAYA ### Thesis assumption jo << j ph,max

                if charge_collection_method:
                    
                    L_E = 350E-4
                    ### OLAYA ### Le: minority carrier diffusion length
                    ### OLAYA ### Where do we find this value. Check this out!
                    
                    S = 80
                    ### OLAYA ### S = Vs (surface recombination velocity) / D (diffusivity)
                    ### OLAYA ### Where do we find this value. Check this out!

                    THICKNESS = 350E-4
                    ### OLAYA ### Photo-absorber thickness
                    
                    THICKNESS_N = 100E-7
                    ### OLAYA ### Photo-absorber thickness - N side
                    ### OLAYA ### Where do we find this value. Check this out!
                    
                    THICKNESS_P = THICKNESS - THICKNESS_N
                    ### OLAYA ### Photo-absorber thickness - P side

                    N_Z = 1000
                    ### OLAYA ### z: spatial coordinate along the depth axis
                    ### OLAYA ### Equation A.2 valid from 0 to 100um. Mirroring equation A.2 from 100 to 350um

                    z = np.linspace(0, THICKNESS, N_Z)
                    ### OLAYA ### star at 0, stop at thickness, step N_Z
                    ### OLAYA ### Equation A.2 valid from 0 to 100um. Mirroring equation A.2 from 100 to 350um

                    G = np.zeros(np.shape(z))
                    ### OLAYA ### matrix filled with 0s: as per the np.shape of z

                    dz = z[1] - z[0]
                    ### OLAYA ### provides the differential values of z

                    for zz in range(N_Z):

                        G[zz] = np.sum(ALPHA_SI * J_photon * np.exp( - ALPHA_SI * z[zz]) * dlabda,0) / 10 ** 4
                        ### OLAYA ### Thesis equation 2.7
                        ### OLAYA ### Since, semiconductor photon absorption is a statistical process, 
                        ### OLAYA ### the charge is not generated uniformly across the depth of the absorber.
                        ### OLAYA ### ALPHA_SI : absorption coefficient of silicon 
                        ### OLAYA ### the charge generation distribution is a function of the depth, z
                        ### OLAYA ### ALPHA_SI : absorption coefficient of silicon 
                        ### OLAYA ### N(labda): incoming photon-density
                        ### OLAYA ### J_photon * dlabda =  N(Labda) ???

                    CP = 1 / (np.cosh((THICKNESS_N - z) / L_E) + np.sinh((THICKNESS_N - z) / L_E) * (np.sinh(z / L_E) 
                            + S * L_E * np.cosh(z / L_E)) / (np.cosh(z / L_E) + S * L_E * np.sinh(z / L_E)))
                    ### OLAYA ### Thesis equation A.2
                    ### OLAYA ### Collection probability 
                    ### OLAYA ### With this equation, only from 0 to 100 um is described
                    ### OLAYA ### the latter halve from 100 to 350 um is found by mirroring Thesis equation A.2

                    CP_2 = 1 / (np.cosh((THICKNESS_P - z) / L_E) + np.sinh((THICKNESS_P - z) / L_E) * (np.sinh(z / L_E) 
                              + S * L_E * np.cosh(z / L_E)) / (np.cosh(z / L_E) + S * L_E * np.sinh(z / L_E)))
                    ### OLAYA ### Thesis equation A.2
                    ### OLAYA ### Collection probability 
                    ### OLAYA ### With this equation, only from 0 to 100 um is described
                    ### OLAYA ### the latter halve from 100 to 350 um is found by mirroring Thesis equation A.2

                    CP_2 = np.flipud((CP_2))
                    ### OLAYA ### np.flipud reverses the order of elements along axis 0 (up/down)
                    ### OLAYA ### the latter halve from 100 to 350 um is found by mirroring Thesis equation A.2 

                    CP[z > THICKNESS_N] = 0
                    ### OLAYA ### If z (depth) > thickness - N side, collection probability (from 0 to 100 um) is 0

                    CP_2[z < THICKNESS_N] = 0
                    ### OLAYA ### If z (depth) < thickness - N side, collection probability (from 100 to 350 um) is 0

                    CP_tot = CP + CP_2
                    ### OLAYA ### total collection probability by adding collection probability on each side

                    current_limit[ii] = Q * np.sum(G[0:N_Z] * CP_tot[0:N_Z] * dz)
                    ### OLAYA ### Thesis equation 2.9
                    ### OLAYA ### Maximum photo-current is found by integrating the product of the two probabilities
                    ### OLAYA ### only pairs that are generated as well as collected, contribute to the photo-current
                             
                # Implementing the Tiedje-Yablonivich method for dark saturation current determination
                ### OLAYA ### LIMITING EFFICIENCY OF SILICON SOLAR CELLS, 
                ### OLAYA ### IEEE TRANSACTIONS ON ELECTRON DEVICES, VOL. ED-31, NO.5, MAY 1984
                
                b1 = (2 / H ** 3 * 1 ** 2 / C ** 2 * E_photon ** 2 * np.exp( - E_photon / K_B / temp_absorber))
                ### OLAYA ### Equation 2 using n=1 from LIMITING EFFICIENCY OF SILICON SOLAR CELLS
                ### OLAYA ### n: index of refraction of the medium. n(air) = 1
                ### OLAYA ### The 1 in the denominator of Bose-Einstein thermal occupation factor has been neglected
                ### OLAYA ### Valid in the limit Eg - u > Kb * T
                ### OLAYA ### u is the internal chemical potential, constant throughout the material
                ### OLAYA ### u equals to the separation between the electron and hole quasi-Fermi levels 
                ### OLAYA ### u equals to solar cell output voltage if ideal contacts and no internal conc. gradient

                integral = np.sum(b1 * ABSORBANCE * dE)
                ### OLAYA ### To simplify the use of the following equation                

                dark_saturation_current = Q * np.pi * integral / 10 ** 4
                ### OLAYA ### Equation 5 from LIMITING EFFICIENCY OF SILICON SOLAR CELLS
                ### OLAYA ### Perfect reflection is assumed: radiative coupling through back surface is neglected
                ### OLAYA ### E-4 for a unit change
                ### OLAYA ### Check this out! Thesis equation 2.4 same as this with solved integral   

                v_oc = (K_B * temp_absorber / Q * np.log(current_limit[ii] / dark_saturation_current))
                ### OLAYA ### Thesis equation A.4
                ### OLAYA ### Thesis assumption jo << j ph,max

        else:

            ABSORBANCE = (LABDA_SPECTRUM * 1E-9 < H * C / E_G[ii])
            ### OLAYA ### Labda: Wavelengths in labda_spectrum are transformed from nm to m
            ### OLAYA ### Thesis section 2.1 equation 2.1: labda_g = H * C / E_G
            ### OLAYA ### labda_g is the photon whavelength corresponding to the band gap energy of the photon absorber
            ### OLAYA ### absobance = labda < labda_g

            REFL = 0
            ### OLAYA ### Assumption: considering that no reflection is happening

            # Initialize v_oc, to converge to it later

            v_oc = E_G[ii]

            E_photon = H * C / (LABDA_SPECTRUM * 1E-9)
            ### OLAYA ### Thesis equation 2.1
            ### OLAYA ### Wavelengths in LABDA_SPECTRUM are transformed from nm to m 

            dE = np.zeros(np.shape(E_photon))
            ### OLAYA ### matrix as per the np.shape of E_photon

            dE[0:-1] = np.abs(E_photon[0:-1] - E_photon[1:])
            ### OLAYA ### provides a matrix with the differential values of labda, element by element

            dE[-1] = dE[-2]
            ### OLAYA ### last value takes the same value of value-before-last-value

            J_photon = POWER_SPECTRUM[0, :] / E_photon * (1 - REFL)
            ### OLAYA ### in functions file: J_photon = INCOMING_SPECTRUM[:, 1] / E_photon
            ### OLAYA ### Power spectrum in [0,:] divided by Ephoton & multiplied for (1_REFL)
            ### OLAYA ### Power spectrum in [0,:] divided by (h c / labda_spectrum) & multiplied by (1-REFL)
            ### OLAYA ### Thesis equation 2.21: Total reflectance = (1-REFL)
            ### OLAYA ### Incoming spectrum for the semiconductor = Incoming spectrum * (1-REFL)
            ### OLAYA ### This final incoming spectrum is the one that should be integrated for the calculations

            dlabda = np.zeros((len(LABDA_SPECTRUM),))
            ### OLAYA ### Matrix Row: 1 & columns: len of labda_spectrum

            dlabda[0:-1] = np.abs(LABDA_SPECTRUM[0:-1] - LABDA_SPECTRUM[1:])
            ### OLAYA ### provides a matrix with the differential values of labda, element by element
           
            dlabda[-1] = dlabda[-2]
            ### OLAYA ### last value takes the same value of value-before-last-value 
     
            # Iterate a few time to converge to a solution

            for i in range(3):         

                current_limit[ii], temp_absorber, input_power[kk], q_in = \
                calculate_flux_balance(INCOMING_SPECTRUM, E_G[ii], T_AIR[kk, 1], ABSORBANCE, T_ELECTROLYTE, v_oc)
                ### OLAYA ### INPUTS: INCOMING_SPECTRUM, E_G, T_AIR, ABSORBANCE, T_ELECTROLYTE, v_oc
                ### OLAYA ### OUTPUT: current_limit, temp_absorber, input_power, q_in

                dark_saturation_current = (Q * A * 2 * K_B * temp_absorber / H ** 3 / C ** 2 * (E_G[ii] ** 2 
                    + 2 * K_B * temp_absorber * E_G[ii] 
                    + 2 * (K_B * temp_absorber) ** 2) * np.exp(- E_G[ii] / K_B / temp_absorber) / 10 ** 4)
                ### OLAYA ### Thesis equation 2.4
                    
                v_oc = (K_B * temp_absorber / Q * np.log(current_limit[ii] / dark_saturation_current))
                ### OLAYA ### Thesis equation A4
                ### OLAYA ### Thesis assumption: jo << jmax,ph

                # Uncomment the next line if you want to fix the temperature

                # temp_absorber = T_ARRAY[kk]
                ### OLAYA ### Assign specific loop value
                ### OLAYA ### In which conditions should be commented / uncommented. Check this out!

        if TEMP_DEPENDENT_J0:

            J0_BV = J0_BV_ref * np.exp(- E_A / R / temp_absorber)
            ### OLAYA ### Thesis equation 2.14
            ### OLAYA ### J0_BV_ref:  exchange current density at a reference temperature
            ### OLAYA ### Ea: Activation Energy which is typically determined experimentally
            ### OLAYA ### R = Kb to match equation
            
        # v_oc = 0.51

        iv_curve[:, ii, :, kk], overpotential = \
        calculate_iv_curve(current_limit[ii], v_oc, J0_BV, temp_absorber, dark_saturation_current, R_SERIES, ALPHA)
        ### OLAYA ### INPUTS: current_limit, v_oc, J0_BV, temp_absorber, dark_saturation_current, R_SERIES, ALPHA
        ### OLAYA ### OUTPUT: iv_curve, overpotential

        soc_correction[ii, :] = (R * temp_absorber / F * np.log(SOC[ii, :] ** 2 / (1 - SOC[ii, :]) ** 2))
        ### OLAYA ### Thesis equation 2.17
        ### OLAYA ### SOC: state of charge between 0 and 1
        ### OLAYA ### higher SOCs result in higher cell voltages 
        ### OLAYA ### which means higher voltages are needed to keep charging beyong 50%
        ### OLAYA ### Vcel - AEo, redox = soc_correction
        ### OLAYA ### Vcell: Cell Voltage. It is 1V at 50% SOC
        ### OLAYA ### AEo,cell: difference in voltage between the two standard redox potentials of the active species
   
    soc_correction[SOC >= 0.999999999] = 1000

    ### OLAYA ### When soc_correction >= 0.999999, soc_correction will be equal to 1000
    ### OLAYA ### 1000? Why these values? Check this out!
    
    for jj in range(NUM_E_THERM):

        condition = numpy.matlib.repmat(E_THERM[jj] + soc_correction[:, jj], NUM_IV_POINTS, 1)
        ### OLAYA ### matlib.repmat(a,m,n): repeat a 0-D to 2-D array or matrix MxN times
        ### OLAYA ### repeats (E_therm[jj] + soc_correction[:,jj]) along the num_IV_points axis and the 1 axis
        ### OLAYA ### creates a matrix with one column, and NUM_IV_POINTS row, 
        ### OLAYA ### each row is the same and equal to ( E_THERM[jj] + soc_correction[:, jj] )
        ### OLAYA ### Check this out!

        condition = np.transpose(condition)
        ### OLAYA ### transpose matrix of condition

        sufficient_photovoltage = iv_curve[0, :, :, kk] > condition
        ### OLAYA ### values that have bigger V than the condition have sufficient photovoltage, selected ones
        ### OLAYA ### values that have an soc_correction >= 0.99999, not included

        operating_current[:, jj] = (np.max(iv_curve[1, :, :, kk] * sufficient_photovoltage, 1))
        ### OLAYA ### Thesis section 2.4, equation 2.26
        ### OLAYA ### Operating current: obtained as the maximum possible current density in the I-V characteristic
        ### OLAYA ### at a voltage that also satisfied the previous condition

        efficiency[:, jj, kk] = (operating_current[:, jj] * (E_THERM[jj] + soc_correction[:, jj]) 
            / input_power[kk] * 100)
        ### OLAYA ### Thesis equation 2.26
        ### OLAYA ### Vcell = ( E_THERM + soc_correction )
        ### OLAYA ### 100 for unit change
        ### OLAYA ### Thesis assumption: overpotential at counter electrode is negligible 
        ### OLAYA ### Thesis assumption: surface area of counter electrode >> photoelectrode

        output_power[:, jj, kk] = (operating_current[:, jj] * (E_THERM[jj] + soc_correction[:, jj]) * 1000)
        ### OLAYA ### Thesis equation 2.26
        ### OLAYA ### Thesis assumption: overpotential at counter electrode is negligible 
        ### OLAYA ### Thesis assumption: surface area of counter electrode >> photoelectrode

        if DYNAMIC_SOC:

            SOC[:, jj] = (SOC[:, jj] + AREA_ABSORBER * operating_current[:, jj] * DT / CAPACITY)
            ### OLAYA ### Thesis equation 2.33
            ### OLAYA ### A: Absorbing area of the semiconductor
            ### OLAYA ### operation_current: operating current of the SRFB (experimentally determined)
            ### OLAYA ### DT: time resolution
            ### OLAYA ### CAPACITY: charge capacity of the battery in coulombs

            SOC[SOC >= 1] = 0.999999999
            ### OLAYA ### SOC: state of charge between 0 and 1
            ### OLAYA ### if SOC >=1, re-assign value to 1 as SOC values are between 0 and 1
 
    dT_electrolyte = q_in / (RHO * VOLUME * C_ELECTROLYTE)
    ### OLAYA ### Thesis equation 2.31
    ### OLAYA ### Transient heat balance, needed to describe the heating and cooling of the electrolyte
    ### OLAYA ### dT_electrolyte: differential of the electrolyte temperature
    ### OLAYA ### q_in: incoming heat fluc for the electrolyte
    ### OLAYA ### RHO: density of the electrolyte 
    ### OLAYA ### VOLUME: volume of the electrolyte
    ### OLAYA ### C_ELECTROLYTE: specific heat of the electrolyte

    # Equilibrate the electrolyte temperature to the morning air temperature if the day changes

    if REAL_SOLAR_DATA:
        
        if int(day_in_the_year[kk + 1] - day_in_the_year[kk]) == 0:   
        ### OLAYA ### when values are the same in the matrix

            T_ELECTROLYTE = T_ELECTROLYTE + dT_electrolyte * DT
            ### OLAYA ### Thesis equation 2.32
        
        elif int(day_in_the_year[kk + 1] - day_in_the_year[kk]) == 1:
        ### OLAYA ### Check this out!

            T_ELECTROLYTE = T_AIR[kk + 1, 1]
            ### OLAYA ### Why is this value re-assigned here ? Check this out!
        
        else:
        ### OLAYA ### Check this out!

            print(int(day_in_the_year[kk + 1] - day_in_the_year[kk]))
            ### OLAYA ### print the value

            T_ELECTROLYTE = T_AIR[kk + 1, 1]
            ### OLAYA ### Why is this value re-assigned here ? Check this out!
    
    output_temperature[kk] = temp_absorber
    ### OLAYA ### temp_absorber is not inside this loop, it is always the same? Check this out!

    output_temperature_electrolyte[kk] = T_ELECTROLYTE

    ### OLAYA ### Next line is commented as per Richard indications:

    # tmp[kk] = overpotential[np.argmin(np.abs(iv_curve[1,:,:,kk] - 0.01))]
    ### OLAYA ### np.argmin: return the indices of the minimum values along an axis
    ### OLAYA ### iv_curve[1,:,:,kk]: current density in the I-V characteristics, I values 
    ### OLAYA ### Overpotential is calculated where the j=0.01 is in the iv curve

    print("\r {}".format(np.round((kk + 1) / NUM_TIME_STEPS * 100)), end = "")    
    ### OLAYA ### why is printing 1/5*100, 2/5*100, 3/5*100, 4/5*100, 5/5*100 if NUM_TIME_STEPS=5 ? Checj this out!

del ii, jj, kk
### OLAYA ### delete COUNT, columns, stripped, OPEN_DATA, LINE

ELAPSED_TIME = time.time() - START_TIME

print('\n The elapsed time = ', ELAPSED_TIME)

test = np.squeeze(efficiency)
### OLAYA ### removes axes of length one from efficiency

### OLAYA ### commented after Richard indications, as no daytime has been created for real_solar_data set to False
#plt.plot(DAYTIME, test)

# #####################################################################################################################
