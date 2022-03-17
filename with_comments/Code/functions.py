
"""
Created on Mon May  6 13:44:07 2019
@author: Richard Faasse
"""

# #####################################################################################################################
# Functions
# #####################################################################################################################

import scipy.optimize as opt
import numpy as np
from device_parameters import C_AUGER, N_I, THICKNESS, H_AIR, H_ELECTROLYTE, H_ELECTROLYTE_AIR, AREA_ABSORBER
from physical_constants import Q, H, K_B, C, SIGMA
from model_settings import REAL_ABSORBER_DATA, AUGER_RECOMBINATION, NUM_IV_POINTS
from losses import JLIM_ANODE, JLIM_CATHODE


# #####################################################################################################################
# Calculate flux balance
# #####################################################################################################################

def calculate_flux_balance(INCOMING_SPECTRUM, E_G, T_AIR, ABSORBANCE, T_ELECTROLYTE, v_oc):
    
    """
    Description:

    This function solves the flux balance based on the incoming solar radiation, 
    black-body emission and convective heat transfer to calculate the limiting photo-current, 
    the temperature of the photo-absorber and the total input power of the incoming spectrum

    INPUTS:

    INCOMING_SPECTRUM: array (N,2): Contains incoming power spectrum in [:,1] and corresponding wavelengths in [:,0]
    E_G: float : Bandgap of the photo-absorber (J)
    T_AIR: float : Temperature of the surrounding air
    ABSORBANCE: array (N,) : absorbance (number between 0 and 1) for every wavelength
    T_ELECTROLYTE: Electrolyte temperature
    ### OLAYA ### v_oc is also an input

    OUTPUTS:

    current_limit: float : limiting photo-current produced by the photoabsorber with bandgap E_G
                            under the influence of the incoming radiation stored in INCOMING_SPECTRUM
    temp_absorber: float : Temperature of the photo-absorber, calculated using the heat balance
    input_power: float : Total input power of the incoming solar spectrum
    q_in: float : Incoming solar flux for the electrolyte
    """

    labda_g = H * C / E_G
    ### OLAYA ### Thesis section 2.1 equation 2.1
    ### OLAYA ### labda_g is the photon whavelength corresponding to the band gap energy of the photon absorber
    
    labda = INCOMING_SPECTRUM[:, 0] * 1E-9
    ### OLAYA ### INCOMING_SPECTRUM[:,0]: wavelength of the incoming power spectrum
    ### OLAYA ### Wavelengths in [:,0] are transformed from nm to m

    E_photon = H * C / labda
    ### OLAYA ### Thesis section 2.1 equation 2.1
   
    J_photon = INCOMING_SPECTRUM[:, 1] / E_photon
    ### OLAYA ### INCOMING_SPECTRUM [:,1]: incoming power spectrum
    ### OLAYA ### Power spectrum in [:,1] divided by Ephoton
    ### OLAYA ### Power spectrum in [:,1] divided by (h c / labda)

    dlabda = np.zeros((len(INCOMING_SPECTRUM),))
    ### OLAYA ### array (row:1 & column: len of incoming_spectrum)

    dlabda[0:-1] = np.abs(INCOMING_SPECTRUM[0:-1, 0] - INCOMING_SPECTRUM[1:, 0])
    ### OLAYA ### provides an array with the differential values of labda, element by element

    dlabda[-1] = dlabda[-2]
    ### OLAYA ### last value takes the same value of value-before-last-value 

    E_dissipated = E_photon - v_oc * Q
    ### OLAYA ### Thesis assumption: only the fraction of the photon energy which corresponds to e*Voc 
    ### OLAYA ### is used to effectively incresase the electron energy while the rest is dissipated into heat
    ### OLAYA ### Of the photon energy Eph, the part Eph-e*Voc is dissipated and fed into te heat balance as phiin

    if REAL_ABSORBER_DATA:

        current_limit = np.sum(J_photon * ABSORBANCE * dlabda) * Q / 10 ** 4
        ### OLAYA ### Thesis equation 2.9
        ### OLAYA ### Thesis N(labda): incoming photon density
        ### OLAYA ### J_photon == N(labda): incoming photon density
        ### OLAYA ### 1E4 unit change between cm^2 and m^2

        solar_heat_flux = np.sum(E_dissipated * J_photon * ABSORBANCE * dlabda) / 10 ** 4
        ### OLAYA ### Thesis equation 2.30
        ### OLAYA ### J_photon == N(labda): incoming photon density
        ### OLAYA ### 1E4 unit change between cm^2 and m^2

    elif not REAL_ABSORBER_DATA:

        current_limit = np.sum(J_photon[labda <= labda_g] * dlabda[labda <= labda_g]) * Q / 10 ** 4
        ### OLAYA ### Thesis equation 2.9
        ### OLAYA ### Thesis N(labda): incoming photon density
        ### OLAYA ### J_photon == N(labda): incoming photon density
        ### OLAYA ### 1E4 unit change between cm^2 and m^2

        solar_heat_flux = np.sum(E_dissipated * J_photon * (labda <= labda_g) * dlabda) / 10 ** 4
        ### OLAYA ### Thesis equation 2.30 
        ### OLAYA ### J_photon == N(labda): incoming photon density
        ### OLAYA ### 1E4 unit change between cm^2 and m^2

    sol = opt.root_scalar(heat_balance, args=(solar_heat_flux, T_AIR, T_ELECTROLYTE), 
        bracket = [200, 600], x0 = 300, method = 'brenth')
    ### OLAYA ### Iteration of heat balance function until a valid solution is found
    ### OLAYA ### Temp_absorber is extracted from the solution of the iteration

    temp_absorber = sol.root
    ### OLAYA ### Assign the solution of the iteration to the temp_absorber 

    q_in = AREA_ABSORBER * (SIGMA * (temp_absorber ** 4 - T_ELECTROLYTE ** 4) + H_ELECTROLYTE_AIR * 
        (T_AIR - T_ELECTROLYTE) + H_ELECTROLYTE * (temp_absorber - T_ELECTROLYTE)) / 10 ** 4
    ### OLAYA ### Thesis equation 2.32
    ### OLAYA ### Thesis equation 2.32 without the second term:(SIGMA * A * (T_AIR ** 4 - T_ELECTROLYTE ** 4))
    ### OLAYA ### Thesis assumption: electrolyte in fully equilibrium with ambient: T_ELECTROLYTE = T_AIR at t=0
    ### OLAYA ### Fourth term: H_ELECTROLYTE * (temp_absorber - T_ELECTROLYTE)
    ### OLAYA ### Fourth term: H_ELECTROLYTE * (temp_absorber - T_ELECTROLYTE) * AREA_AIR_ELECTROLYTE 
    ### OLAYA ### Fourth term: incorrect unless AREA_ABSORBER == AREA_AIR_ABSORBER
    
    input_power = np.sum(E_photon * dlabda * J_photon) / 10 ** 4
    ### OLAYA ### J_photon == N(labda): incoming photon density
    ### OLAYA ### E_photon: photon energy     
    ### OLAYA ### 1E4 unit change between cm^2 and m^2
    ### OLAYA ### this step should be equal to: input_power = np.sum(INCOMING_SPECTRUM[:,1]) / 10 ** 4

    return current_limit, temp_absorber, input_power, q_in
  
# #####################################################################################################################
# Calculate heat balance
# #####################################################################################################################

def heat_balance(temp_absorber, solar_heat_flux, T_AIR, T_ELECTROLYTE):

    """
    Description:

    This function describes the heat balance, using the incoming solar heat flux, 
    as calculated in calculate_heat_balance and the respective temperatures of the absorber,
    electrolyte and ambient air and the convective heat transfer coefficients (H_AIR/ELECTROLYTE etc)

    INPUTS:

    temp_absorber: float : Temperature of the photo-absorber
    solar_heat_flux: float : the dissipated incoming solar radiation
    T_AIR: float : Temperature of the surrounding air
    T_ELECTROLYTE: Electrolyte temperature

    OUTPUTS:

    y: float: output of the function that should be 0 in the assumed steady state heat balance, 
                such that it is used for root finding in the function calculate_heat_balance
    """

    y = (- solar_heat_flux + SIGMA * (temp_absorber ** 4 - T_AIR ** 4) / 10 ** 4
         + H_AIR * (temp_absorber - T_AIR) / 10 ** 4 + H_ELECTROLYTE * (temp_absorber - T_ELECTROLYTE) / 10 ** 4)
    ### OLAYA ### Thesis equation 2.29 
    ### OLAYA ### Thesis equation 2.29 without the first term: (SIGMA * (temp_absorber ** 4 - T_ELECTROLYTE ** 4))
    ### OLAYA ### Thesis assumption: electrolyte in fully equilibrium with ambient: T_ELECTROLYTE = T_AIR at t=0
    
    return y

# #####################################################################################################################
# Calculate Butler Volmer
# #####################################################################################################################

def butler_volmer(overpotential, J0_BV, temp_absorber, j_fit, ALPHA):
    
    """
    Description:

    This function describes the heat balance, using the incoming solar heat flux, 
    as calculated in calculate_heat_balance and the respective temperatures of the absorber, 
    electrolyte and ambient air and the convective heat transfer coefficients (H_AIR/ELECTROLYTE etc)

    INPUTS:

    overpotential: float : overpotential, which needs to be solved
    J0_BV: float : Exchange current density as defined in the butler-volmer equation
    temp_absorber: float : Temperature of the photo-absorber
    j_fit: float : current density at which we want to solve for V
    ALPHA: float : asimmetry factor from the butler-volmer equation

    OUTPUTS:

    y: float : output of the function that should be 0 to solve for the overpotential, 
                such that it is used for root finding in the function calculate_iv_curve
    """

    y = J0_BV * (np.exp(ALPHA * Q * overpotential / K_B / temp_absorber) - 
        np.exp( - (1 - ALPHA) * Q * overpotential / K_B / temp_absorber)) - j_fit
    ### OLAYA ### Thesis equation 2.13
    ### OLAYA ### J0_BV is the exchange current density
    ### OLAYA ### ALPHA is the transfer coefficient, accounts for the symmetry of the kinetic overpotential losses
    ### OLAYA ### overpotential = (E - Eo) 
    ### OLAYA ### n = 1  assuming ideality
    ### OLAYA ### (- j_fit) it corresponds with the current density at which we solve for V 
    ### OLAYA ### V is the solucion of the equation
  
    # y = J0_BV*((1-j_fit/JLIM_ANODE) * np.exp(ALPHA * Q * overpotential / K_B / temp_absorber) 
    #    - (1-j_fit/JLIM_CATHODE) * np.exp(-(1-ALPHA)*Q*overpotential/K_B/temp_absorber)) - j_fit
    ### OLAYA ### Thesis equation 2.15
    ### OLAYA ### As previous equation adding (1-j_fit/JLIM_ANODE) and (1-j_fit/JLIM_CATHODE)
    ### OLAYA ### To include mass transfer limitations 
    ### OLAYA ### which account for finite average time it takes for convertable redo species to reach the electrode
    ### OLAYA ### JLIM_ANODE/CATHODE are the mass transfer limited maximum current densities for the anode/cathode

    return y

# #####################################################################################################################
# Calculate IV with Auger
# #####################################################################################################################

def IV_with_Auger(V, dark_saturation_current, current_limit, j, temp_absorber):
    
    """
    Description:

    This function describes the heat balance, using the incoming solar heat flux, 
    as calculated in calculate_heat_balance and the respective temperatures of the absorber, 
    electrolyte and ambient air and the convective heat transfer coefficients (H_AIR/ELECTROLYTE etc)

    INPUTS:

    V: float : voltage, which needs to be solved
    dark_saturation_current: float : dark saturation current of the photo-absober
    current_limit: float : Saturation current of the photo-diode
    j: float: current density at which we want to solve for V
    temp_absorber: float : Temperature of the photo-absorber

    OUTPUTS:

    y: float : output of the function that should be 0 to solve for the voltage, 
                such that it is used for root finding in the function calculate_iv_curve
    """

    x = V * Q / K_B / temp_absorber
    ### OLAYA ### just for simplifying the following equation

    y = ( - dark_saturation_current * np.exp(x) - Q * THICKNESS * C_AUGER * N_I ** 3 * np.exp(3 * x / 2) 
        + current_limit - j)
    ### OLAYA ### Thesis equation 2.6 for solving the V at the current density j

    return y


# #####################################################################################################################
# Calculate IV curve 
# #####################################################################################################################

def calculate_iv_curve(current_limit, v_oc, J0_BV, temp_absorber, dark_saturation_current, R_SERIES, ALPHA):
    
    """
    Description:

    This function calculates the IV-curve of the PEC-flow battery, using various loss inputs and device parameters

    INPUTS:

    current_limit: float : Saturation current of the photo-diode
    v_oc: float: open circuit voltage as calculated
    J0_BV: float : Exchange current density as defined in the butler-volmer equation
    temp_absorber: float: Temperature of the photo-absorber
    dark_saturation_current: float: dark saturation current of the photo-absober
    R_SERIES: float: Total series resistance of the SRFB
    ALPHA: float: Asymmetry factor in the butler volmer equation.
   
    OUTPUTS:
    iv_curve: array (2, NUM_IV_POINTS): current-voltage characteristic, 
            current is found in iv_curve[1,:], voltage is found in iv_curve[0,:]
    overpotential: array (NUM_IV_POINTS,): kinetic overpotential as a function of current 
            (the current is found in j_fit, or iv_curve[1,:])
    """

    j_fit = - np.linspace(0 * 0.999999 * current_limit, current_limit, NUM_IV_POINTS)
    ### OLAYA ### start at 0 * 0.999999 * current_limit ( = 0), finish at current limit, step: num_IV_points
    ### OLAYA ### shape is (1000, 1) and size is 1000

    # j_fit = -np.linspace(0.9999*JLIM_CATHODE, 0.9999*JLIM_ANODE, NUM_IV_POINTS)
    ### OLAYA ### start at 0 * 0.999999 * current_limit, finish at current limit, step: num_IV_points
    ### OLAYA ### Shape is (1000,) and size is 1000

    j_D = current_limit + j_fit
    ### OLAYA ### j_D is the current through the diode which is equal to jmax,ph - j
    ### OLAYA ### Thesis equation 2.3
    ### OLAYA ### Thesis equation 2.24
    ### OLAYA ### As per the Shockley diode equation & Kirchoff's Law: j_fit = current_limit - j_D
    ### OLAYA ### if j_fit = current_limit - j_D ---> j_D = current_limit - j_fit
    ### OLAYA ### j_fit defied with a - before. it should be okei

    V_D = np.zeros(np.shape(j_fit))

    j_D[j_D == 0] = 1E-15
    ### OLAYA ### Why this value ? Just a small value

    if ALPHA == 0.5:

        overpotential = 2 * K_B * temp_absorber / Q * np.arcsinh(j_fit / 2 / J0_BV)
        ### OLAYA ### Thesis equation 2.23

    else:

        overpotential = np.zeros((np.size(j_fit, 0), ))

        for ii in range(np.size(j_fit, 0)):

            sol = opt.root_scalar(butler_volmer, args = (J0_BV, temp_absorber, j_fit[ii], ALPHA), 
                bracket = [-10, 10], x0 = 0, method = 'brenth')
            ### OLAYA ### Iteration of butler volmer function until a valid solution is found
            ### OLAYA ### Overpotential is extracted from the solution of the iteration

            overpotential[ii] = sol.root
            ### OLAYA ### Assign the solution of the iteration to that position on the overpotential array

    if AUGER_RECOMBINATION:

        for ii in range(np.size(j_fit, 0)):

            sol = opt.root_scalar(IV_with_Auger, args = (dark_saturation_current, current_limit, 
                - j_fit[ii], temp_absorber), bracket = [- 1000, 1000], x0 = 0, method = 'brenth')
            ### OLAYA ### Iteration of IV_with_AUGER function until a valid solution is found
            ### OLAYA ### V at the current density j is extracted from the solution of the iteration

            V_D[ii] = sol.root
            ### OLAYA ### Assign the solution of the iteration to that position on the V_D array

    else:

        V_D = v_oc + K_B * temp_absorber / Q * np.log((j_D) / (current_limit))
        ### OLAYA ### Thesis equation A.3
        ### OLAYA ### Thesis equation 2.24 uses jo which should correspond with the dark current. 
        ### OLAYA ### Thesis equation A.3 uses jmax,ph which should correspond with current limit
        ### OLAYA ### The assumption used to get to the equation is jmax,ph >> jo
        ### OLAYA ### for that reason, thesis equation A.3 looks to be rigth and 2.24 looks to be wrong
      
    PV = V_D + overpotential + j_fit * R_SERIES
    ### OLAYA ### Thesis equation 2.22
    ### OLAYA ### PV = Vc - E0
    ### OLAYA ### Vc(j) circuit voltage: voltage between the two open nodes
    ### OLAYA ### overpotential = E - Eo
    ### OLAYA ### VR = j * R_SERIES voltage drop due to the series resistance
    ### OLAYA ### PV = V_D + overpotential + VR

    # j_fit[np.isnan(PV)] = - current_limit

    j_fit = np.abs (j_fit)
 
    iv_curve = np.squeeze(np.array([[PV], [j_fit]]))

    return iv_curve, overpotential
   
# #####################################################################################################################

