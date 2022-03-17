# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:43:31 2019

@author: Richard Faasse
"""

from physical_constants import N_A, Q
import numpy as np


# =============================================================================
# Define the device parameters
# =============================================================================

AREA_ABSORBER = 0.25                            # Area absorber:                             [cm2]
    
AREA_ELECTROLYTE = 1E4                          # Area between the electrolyte and the air   [cm2]
                                        
VOLUME = 20E-6                                  # Volume electrolyte:                        [m3]

CAPACITY = 0.4 * VOLUME * 1000* N_A *Q / 10000  # Charge capacity battery                    [C]

SOC = 0.5                                       # State-of-charge battery                    [-]

A = 2


# =============================================================================
# Heat exchange parameters
# =============================================================================

H_AIR = 10

H_ELECTROLYTE = 1000

H_ELECTROLYTE_AIR = 10 * AREA_ELECTROLYTE / AREA_ABSORBER

RHO = 998

C_ELECTROLYTE = 4200


# =============================================================================
# Silicon specific parameters
# =============================================================================

C_AUGER = 3.88E-31

N_I = 1.45E10

THICKNESS = 350E-4                              # Thickness of absorber                      [cm]
