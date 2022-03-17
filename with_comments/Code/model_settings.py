
"""
Created on Mon May  6 13:44:07 2019
@author: Richard Faasse
"""

# #####################################################################################################################
# Define the model settings
# #####################################################################################################################

REAL_SOLAR_DATA = True             # True if you want real NREL data, in that case fill in a date in simulation.py, 
                                    # False if you want the AM15 spectrum

REAL_ABSORBER_DATA = True          # Choose if you want to use the real silicon absorption spectra, 
                                    # or just absorption calculated using the band-gap

DYNAMIC_SOC = True                 # Choose if you want an SOC that changes during the day


TEMP_DEPENDENT_J0 = True           # If you want a temperature dependent J0, 
                                    # check whether the J0_BV_ref and E_a are well defined

AUGER_RECOMBINATION = True         # Using Auger recombination is only accurate 
                                    # if REAL_ABSORBER_DATA = True

save_fit_progression = True        # Choose True if you want to save the parameters that
                                    # the fit-tool varies throughout the fitting process
                                  
charge_collection_method = True    # Choose True if you want to use the charge collection method, 
                                    # False if you don't


# #####################################################################################################################
# Define the resolutions of the model inputs
# #####################################################################################################################

NUM_IV_POINTS = 1000                # Number of datapoints used for the iv-curve


NUM_E_G = 100                       # Number of band-gap energies 
                                    # It will be changed to 1 if REAL_ABSORBER_DATA = True

NUM_E_THERM = 100                   # Number of thermodynamic potentials,
                                    # set the range in simulation.py

NUM_TIME_STEPS = 1                  # Set number of time-steps in case of parameter dependent simulation


# #####################################################################################################################

