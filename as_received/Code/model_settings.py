# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:44:43 2019

@author: Richard
"""

# =============================================================================
# Define the settings
# =============================================================================

REAL_SOLAR_DATA = False            # True if you want real NREL data, in that 
                                  # case fill in a date in simulation.py, False
                                  # if you want the AM15 spectrum

REAL_ABSORBER_DATA = False        # Choose if you want to use the real silicon
                                  # Absorption spectra, or just absorption cal-
                                  # culated using the band-gap

DYNAMIC_SOC = False               # Choose if you want an SOC that changes du-
                                  # ring the day

TEMP_DEPENDENT_J0 = False         # If you want a temperature dependent J0, 
                                  # check whether the J0_BV_ref and E_a are 
                                  # well defined

AUGER_RECOMBINATION = False       # Using Auger recombination is only accurate 
                                  # if REAL_ABSORBER_DATA = True

save_fit_progression = False      # Choose True if you want to save the para-
                                  # meters the fit-tool varies throughout the
                                  # fitting process
                                  
charge_collection_method = False  # Choose True if you want to use the charge
                                  # collection method, False if you don't


# =============================================================================
# Define the resolutions of the model inputs
# =============================================================================

NUM_IV_POINTS = 1000              # Number of datapoints used for the iv-curve

NUM_E_G = 100                     # Number of band-gap energies (will be
                                  # changed to 1 if REAL_ABSORBER_DATA = True)

NUM_E_THERM = 100                 # Number of thermodynamic potentials, set
                                  # the range in simulation.py

NUM_TIME_STEPS = 1                # Set number of time-steps in case of para-
                                  # meter dependent simulation
