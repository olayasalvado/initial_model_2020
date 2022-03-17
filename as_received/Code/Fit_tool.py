# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:16:21 2019

@author: Richard Faasse
"""

import scipy.optimize as opt
import time
import numpy as np
import matplotlib.pyplot as plt
from functions import butler_volmer
from physical_constants import Q, K_B
from model_settings import save_fit_progression
import sys

#sys.path.insert(0,'/media/sf_python/initial_model_2020/as_received/input_data/')
input_data_path = '/media/sf_python/initial_model_2020/as_received/input_data/'

#%% Load data
#exp_IV = np.load('BPA_ferro.npy')
#exp_IV = -exp_IV[1000:1620, :]

#exp_IV = np.transpose(np.load('CV_20190514.npy'))
#exp_IV = np.load('BPC_PtSi.npy')
#exp_IV = np.load('50SOC_SiC_lowSS.npy')
#exp_IV = np.transpose(np.load('CV_20180417_sheet3.npy'))

exp_IV = np.load(input_data_path + 'BPC_ferri.npy')
exp_IV = exp_IV[200:1000, :]

exp_IV[:, 1] = exp_IV[:, 1] / 1000

#plt.plot(exp_IV[:, 0], exp_IV[:, 1])
#plt.show()


#%%
if save_fit_progression:
    fit_progression = np.zeros((3000, 3))
    np.save('fit_progression.npy', fit_progression)

JLIM_ANODE = 0.1
JLIM_CATHODE = - 0.03
temp_absorber = 300
V_oc = exp_IV[np.argmin(np.abs(exp_IV[:, 1])), 0]
current_limit = 0.04

def IV_error(x):
    """
    Description:
    ------------
    This function calculates error between experimental data and modeled curves
    using three parameters: ALPHA, R_SERIES and J0_BV, taken together in the 
    array x

    INPUTS:
    ------------
    x: array (3,)
        Array containing ALPHA, R_SERIES and J0_BV as variable parameters

    OUTPUTS:
    ----------
    error_IV: float
        average squared error between the experimental data and modeled curve
    """
    
    
    ALPHA = x[0]
    R_SERIES = x[1]
    J0_BV = 10 ** (x[2])

    j_fit = np.linspace(JLIM_CATHODE + 0.00001, JLIM_ANODE-0.00001, 1000)
    overpotential = np.zeros((np.size(j_fit, 0), ))
    error_small_enough = np.zeros((np.size(j_fit, 0)), dtype = bool)
    error = np.zeros((np.size(j_fit, 0), ))
    j_D = current_limit + j_fit


    # Find the overpotential as function of current density
    for ii in range(np.size(j_fit, 0)):
        sol = opt.root_scalar(butler_volmer, args = (J0_BV, temp_absorber, j_fit[ii], ALPHA), bracket = [-10, 10], x0 = 0, method = 'brenth')
        overpotential[ii] = sol.root

        # Check whether overpotential solution is correct:
        error[ii] = np.abs(butler_volmer(overpotential[ii], J0_BV, temp_absorber, j_fit[ii], ALPHA))

        error_small_enough[ii] = error[ii] < 10 ** (- 8)

    V_D = V_oc + K_B * temp_absorber / Q * np.log((j_D) / (current_limit))
    PV = V_D + overpotential + j_fit*R_SERIES

    j_interp = np.interp(exp_IV[:, 0], PV[error_small_enough == 1], j_fit[error_small_enough == 1])
    j_interp[np.isnan(j_interp)] = - current_limit


    error_IV = (np.sum(np.abs(np.power(1000 * exp_IV[:, 1] - 1000 * j_interp, 2))))

    print('total squared error = ', str.format('{0:.2f}', np.sqrt(error_IV)),'mA/cm2, ALPHA = ', str.format('{0:.2f}', ALPHA),
          ', R_SERIES = ', str.format('{0:.1f}', R_SERIES),', J0_BV = 10 ^ ', str.format('{0:.1f}', x[2]))

    if save_fit_progression:
        fit_progression = np.load('fit_progression.npy')
        fit_progression[np.sum(fit_progression[:, 0] > 0), :] = np.array((ALPHA, R_SERIES, J0_BV))
        np.save('fit_progression.npy', fit_progression)
    return error_IV

START_TIME = time.time()


bounds = [(0.1, 0.9), (0, 40), (-8, 0.5)]
x = opt.dual_annealing(IV_error, bounds, initial_temp = 100, maxfun = 100)
ELAPSED_TIME = time.time() - START_TIME

fitted_parameters = x.x


#%% Plot the modeled and experimental curves

ALPHA = fitted_parameters[0]
R_SERIES = fitted_parameters[1]
J0_BV = 10 ** (fitted_parameters[2])


j_fit = np.linspace(JLIM_CATHODE+0.00001,JLIM_ANODE-0.00001, 1000)
overpotential = np.zeros((np.size(j_fit, 0), ))
error_small_enough = np.zeros((np.size(j_fit, 0)), dtype = bool)
error = np.zeros((np.size(j_fit, 0), ))
j_D = current_limit + j_fit


# Find the overpotential as function of current density
for ii in range(np.size(j_fit, 0)):
    sol = opt.root_scalar(butler_volmer, args = (J0_BV, temp_absorber, j_fit[ii], ALPHA), bracket = [-10, 10], x0 = 0, method = 'brenth')
    overpotential[ii] = sol.root

    # Check whether overpotential solution is correct:
    error[ii] = np.abs(butler_volmer(overpotential[ii], J0_BV, temp_absorber, 
                                     j_fit[ii], ALPHA))

    error_small_enough[ii] = error[ii] < 10 ** (- 8)

V_D = V_oc + K_B * temp_absorber / Q * np.log((j_D) / (current_limit))
PV = V_D + overpotential + j_fit * R_SERIES

j_interp = np.interp(exp_IV[:, 0], PV[error_small_enough == 1],j_fit[error_small_enough == 1])
j_interp[np.isnan(j_interp)] = -current_limit



error_IV = np.sum(np.abs(exp_IV[:, 1] - j_interp)) / np.size(exp_IV, 0) * 1000
print('The elapsed time = ', ELAPSED_TIME)
print('average error (mA/cm2) = ', error_IV)
print('Alpha = ', ALPHA)
print('R_SERIES = ', R_SERIES)
print('J0_BV = ', J0_BV )
print('V_oc = ', abs(V_oc), 'vs NHE')
fig,ax = plt.subplots()

plt.plot(exp_IV[:, 0],exp_IV[:, 1]*1000, label = 'experimental data', linewidth = 2, color = (0,166 / 255,214 / 255))
plt.plot(exp_IV[:,0],j_interp*1000, linestyle = '--', label = 'fit', color = 'black', linewidth = 2)
   
ax.spines['bottom'].set_position('zero')

plt.legend()
plt.ylabel('Current density (mA/cm$^2$)')
plt.xlabel('V vs NHE (V)')
plt.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.savefig('fit_BPC_PEC.svg', format = 'svg')
plt.show()

    
