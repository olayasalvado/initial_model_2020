
"""
Created on Mon May  6 13:44:07 2019
@author: Richard Faasse
"""

# #####################################################################################################################
# Define the losses
# #####################################################################################################################

R_SERIES = 0                # Includes both PV resistance as well as electrolyte/membrane resistance     [Ohm*cm2]
                         
JLIM_ANODE = 1000000        # Mass transfer limited anodic current                                       [mA/cm2]    

JLIM_CATHODE = -1000000     # Mass transfer limited cathodic current                                     [mA/cm2]

J0_BV = 0.1                 # Exchange current density (bv equation)                                     [mA/cm2]

J0_BV_ref = 142020          # Reference exchange current density for temperature dependence              [mA/cm2]

E_A = 28900

ALPHA = 0.5                 # Symmetry factor bv equation                                                [-]


# #####################################################################################################################

