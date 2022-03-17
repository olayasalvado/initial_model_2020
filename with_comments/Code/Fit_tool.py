
"""
Created on Mon May  6 13:44:07 2019
@author: Richard Faasse
"""

# #####################################################################################################################
# Fit tool
# #####################################################################################################################

import scipy.optimize as opt
import time
import numpy as np
import matplotlib.pyplot as plt
from functions import butler_volmer
from physical_constants import Q, K_B
from model_settings import save_fit_progression
import sys

#sys.path.insert(0,'/media/sf_python/initia_model_2020/with_comments/input_data/')
input_data_path = '/media/sf_python/initia_model_2020/with_comments/input_data/'

# #####################################################################################################################
# Load data
# #####################################################################################################################

#exp_IV = np.load('BPA_ferro.npy')

#exp_IV = - exp_IV[1000:1620, :]

#exp_IV = np.transpose(np.load('CV_20190514.npy'))

#exp_IV = np.load('BPC_PtSi.npy')

#exp_IV = np.load('50SOC_SiC_lowSS.npy')

#exp_IV = np.transpose(np.load('CV_20180417_sheet3.npy'))

exp_IV = np.load(input_data_path + 'BPC_ferri.npy')

exp_IV = exp_IV[200:1000, :]
### OLAYA ### Limits due to experimental limits

exp_IV[:, 1] = exp_IV[:, 1] / 1000
### OLAYA ### Change of units

#plt.plot(exp_IV[:, 0], exp_IV[:, 1])
### OLAYA ### exp_IV[:,0]: Experimental I-V values from laboratory, selecting the column of V values. 
### OLAYA ### exp_IV[:,1]: Experimental I-V values from laboratory, selecting the column of I values. 

#plt.show()

# #####################################################################################################################

if save_fit_progression:

    fit_progression = np.zeros((3000, 3))
    ### OLAYA ### Three columns for (ALPHA, R_SERIES, J0_BV)

    np.save('fit_progression.npy', fit_progression)
   
JLIM_ANODE = 0.1
### OLAYA ### Mass transfer limited maximum current density for the anode
### OLAYA ### Why this value? Check this out!

JLIM_CATHODE = - 0.03
### OLAYA ### Mass transfer limited maximum current density for the anode
### OLAYA ### Why this value? Check this out!

temp_absorber = 300

V_oc = exp_IV[np.argmin(np.abs(exp_IV[:, 1])), 0]
### OLAYA ### returns the indices of the minimum values of (exp_IV[:,1] in absolute value) along the axis 0
### OLAYA ### exp_IV[:,1]: Experimental I-V values from laboratory, selecting the column of I values. 
### OLAYA ### axis 0 on exp_IV should correspond with V values
### OLAYA ### indice of the minimum values of the column I values, along the axis 0

current_limit = 0.04
### OLAYA ### Why this value? Check this out?

# #####################################################################################################################

def IV_error(x):
    
    """
    Description:
    ------------
    This function calculates error between experimental data and modeled curves using three parameters: 
    ALPHA, R_SERIES and J0_BV, taken together in the array x

    INPUTS:
    ------------
    x: array (3,): Array containing ALPHA, R_SERIES and J0_BV as variable parameters
                    ALPHA: float : asimmetry factor from the butler-volmer equation
                    J0_BV: float : Exchange current density as defined in the butler-volmer equation
                    R_SERIES: float: Total series resistance of the SRFB

    OUTPUTS:
    ----------
    error_IV: float: average squared error between the experimental data and modeled curve
    """
    
    ALPHA = x[0]
    ### OLAYA ### Thesis section 2.4.1

    R_SERIES = x[1]
    ### OLAYA ### Thesis section 2.4.1

    J0_BV = 10 ** (x[2])
    ### OLAYA ### Thesis section 2.4.1

    j_fit = np.linspace(JLIM_CATHODE + 0.00001, JLIM_ANODE - 0.00001, 1000)
    ### OLAYA ### j_fit is a float that represents the current density 
    ### OLAYA ###  +- 0.00001 ensures that JLIM_CATHODE/ANODE are both included on the current range 

    overpotential = np.zeros((np.size(j_fit, 0), ))
    ### OLAYA ### array (row:1 & column: size of j_fit at axis 0)

    error_small_enough = np.zeros((np.size(j_fit, 0)), dtype = bool)
    ### OLAYA ### array (row:1 & column: size of j_fit at axis 0)

    error = np.zeros((np.size(j_fit, 0), ))
    ### OLAYA ### array (row:1 & column: size of j_fit at axis 0)
    
    j_D = current_limit + j_fit
    ### OLAYA ### j_D is the current through the diode which is equal to jmax,ph - j
    ### OLAYA ### Thesis equation 2.3
    ### OLAYA ### Thesis equation 2.24
    ### OLAYA ### As per the Shockley diode equation & Kirchoff's Law: j_fit = current_limit - j_D
    ### OLAYA ### if j_fit = current_limit - j_D ---> j_D = current_limit - j_fit
    ### OLAYA ### Why + instead of -? 

    # Find the overpotential as function of current density

    for ii in range(np.size(j_fit, 0)):

        sol = opt.root_scalar(butler_volmer, args = (J0_BV, temp_absorber, j_fit[ii], ALPHA), 
            bracket = [-10, 10], x0 = 0, method = 'brenth')
        ### OLAYA ### Iteration of butler volmer function until a valid solution is found
        ### OLAYA ### Overpotential is extracted from the solution of the iteration

        overpotential[ii] = sol.root
        ### OLAYA ### Assign the solution of the iteration to that position on the overpotential array 

        # Check whether overpotential solution is correct:

        error[ii] = np.abs(butler_volmer(overpotential[ii], J0_BV, temp_absorber, j_fit[ii], ALPHA))
        ### OLAYA ### runs the butler volmer function with the obtained value for the ovepotential  

        error_small_enough[ii] = error[ii] < 10 ** (- 8)
        ### OLAYA ### Check if error is smaller than E-8
        ### OLAYA ### Solution should be a boolean

    V_D = V_oc + K_B * temp_absorber / Q * np.log((j_D) / (current_limit))
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

    j_interp = np.interp(exp_IV[:, 0], PV[error_small_enough == 1], j_fit[error_small_enough == 1])
    ### OLAYA ### "one-dimensional linear interpolation for monotonically increasing sample points"
    ### OLAYA ### obtains j values for V values in exp_IV from values on j_fit 

    j_interp[np.isnan(j_interp)] = - current_limit
    ### OLAYA ### np.isnan: Test element-wise for NaN and return result as a boolean array
    ### OLAYA ### selects just that values that are not empty

    error_IV = (np.sum(np.abs(np.power(1000 * exp_IV[:, 1] - 1000 * j_interp, 2))))
    ### OLAYA ### Thesis equation 2.27 
    ### OLAYA ### exp_IV[:,1]: Experimental I-V values from laboratory, selecting the column of I values. 
   
    print('total squared error = ', str.format('{0:.2f}', np.sqrt(error_IV)),'mA/cm2, ALPHA = ', str.format('{0:.2f}',
     ALPHA), ', R_SERIES = ', str.format('{0:.1f}', R_SERIES),', J0_BV = 10 ^ ', str.format('{0:.1f}', x[2]))
    
    if save_fit_progression:

        fit_progression = np.load('fit_progression.npy')
        
        fit_progression[np.sum(fit_progression[:, 0] > 0), :] = np.array((ALPHA, R_SERIES, J0_BV))
        ### OLAYA ### Assign values which are > 0 to (ALPHA, R_SERIES, J0_BV) 

        np.save('fit_progression.npy', fit_progression)

    return error_IV

START_TIME = time.time()

bounds = [(0.1, 0.9), (0, 40), (-8, 0.5)]
### OLAYA ### Why these values? Check this out?
### OLAYA ### bounds for variables. (min, max) pairs for each element in x: (ALPHA, R_SERIES, J0_BV)

x = opt.dual_annealing(IV_error, bounds, initial_temp = 100, maxfun = 100)
### OLAYA ### opt.dual_annealing: find the global minimum of a function using Dual Annealing
### OLAYA ### IV_error is the function to be minimized / optimizated 
### OLAYA ### resultant x is an array ([ALPHA, R_SERIES, J0_BV])

ELAPSED_TIME = time.time() - START_TIME

fitted_parameters = x.x
### OLAYA ### X attribute: Variable value in the current solution.

# #####################################################################################################################
# Plot the modeled and experimental curves
# #####################################################################################################################

ALPHA = fitted_parameters[0]

R_SERIES = fitted_parameters[1]

J0_BV = 10 ** (fitted_parameters[2])

j_fit = np.linspace(JLIM_CATHODE + 0.00001, JLIM_ANODE - 0.00001, 1000)
### OLAYA ### Assumption: +- 0.00001 ensures that JLIM_CATHODE/ANODE are both included on the current range

overpotential = np.zeros((np.size(j_fit, 0), ))
### OLAYA ### array filled with 0s: row:1 & column: size of j_fit at axis 0

error_small_enough = np.zeros((np.size(j_fit, 0)), dtype = bool) 
### OLAYA ### array (row:1 & column: size of j_fit at axis 0)

error = np.zeros((np.size(j_fit, 0), ))
### OLAYA ### array (row:1 & column: size of j_fit at axis 0)

j_D = current_limit + j_fit
### OLAYA ### j_D is the current through the diode which is equal to jmax,ph - j
### OLAYA ### Thesis equation 2.3
### OLAYA ### Thesis equation 2.24
### OLAYA ### As per the Shockley diode equation & Kirchoff's Law: j_fit = current_limit - j_D
### OLAYA ### if j_fit = current_limit - j_D ---> j_D = current_limit - j_fit
### OLAYA ### Why + instead of -? 

# #####################################################################################################################
# Find the overpotential as function of current density
# #####################################################################################################################

for ii in range(np.size(j_fit, 0)):

    sol = opt.root_scalar(butler_volmer, args = (J0_BV, temp_absorber, j_fit[ii], ALPHA), 
        bracket = [-10, 10], x0 = 0, method = 'brenth')
     ### OLAYA ### Iteration of butler volmer function until a valid solution is found
    ### OLAYA ### Overpotential is extracted from the solution of the iteration

    overpotential[ii] = sol.root
    ### OLAYA ### Assign the solution of the iteration to that position on the overpotential array 

    # Check whether overpotential solution is correct:

    error[ii] = np.abs(butler_volmer(overpotential[ii], J0_BV, temp_absorber, j_fit[ii], ALPHA))
    ### OLAYA ### runs the butler volmer function with the obtained value for the ovepotential  

    error_small_enough[ii] = error[ii] < 10 ** (- 8)
    ### OLAYA ### check if error is inferior to 10E-8
    ### OLAYA ### Solution should be a boolean

V_D = V_oc + K_B * temp_absorber / Q * np.log((j_D) / (current_limit))
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

j_interp = np.interp(exp_IV[:, 0], PV[error_small_enough == 1], j_fit[error_small_enough == 1])
### OLAYA ### "one-dimensional linear interpolation for monotonically increasing sample points"
### OLAYA ### obtains j values for V values in exp_IV from values on j_fit

j_interp[np.isnan(j_interp)] = - current_limit
### OLAYA ### selects just that values that are not empty, in case any of them do not have a value

# #####################################################################################################################

error_IV = np.sum(np.abs(exp_IV[:, 1] - j_interp)) / np.size(exp_IV, 0) * 1000
### OLAYA ### Thesis equation 2.28. 
### OLAYA ### exp_IV[:,1]: Experimental I-V values from laboratory, selecting the column of I values. 

print('The elapsed time = ', ELAPSED_TIME)
print('average error (mA/cm2) = ', error_IV)
print('Alpha = ', ALPHA)
print('R_SERIES = ', R_SERIES)
print('J0_BV = ', J0_BV )
print('V_oc = ', abs(V_oc), 'vs NHE')

fig,ax = plt.subplots()

plt.plot(exp_IV[:,0],exp_IV[:, 1] * 1000, label = 'experimental data', linewidth = 2, color = (0,166 / 255,214 / 255))

plt.plot(exp_IV[:,0],j_interp * 1000, linestyle = '--', label = 'fit', color = 'black', linewidth = 2)
   
ax.spines['bottom'].set_position('zero')

plt.legend()

plt.ylabel('Current density (mA/cm$^2$)')

plt.xlabel('V vs NHE (V)')

plt.tight_layout()

plt.rcParams.update({'font.size': 14})

plt.savefig('fit_BPC_PEC.svg', format = 'svg')

plt.show()

# #####################################################################################################################
