#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:26:04 2021

@author: hauke
"""

from collections import namedtuple

# Time management
WARMUP = 500
# Duration of the simulation (in ms)
sim_time = 10800. # ms
sim_time = 2000. # ms

# Transfer function (sigmoid)
STEEPNESS = 1
X_OFFSET = 10


# Population
Setup = namedtuple("Setup", ("nrows", "NE", "NI", "inh_strength", "exc_strength"))
grid80q = Setup(nrows=80, NE=80**2, NI=40**2, inh_strength=-22, exc_strength=4)
grid60q = Setup(nrows=60, NE=60**2, NI=30**2, inh_strength=-22, exc_strength=4)
# grid60 = Setup(nrows=60, NE=60**2, NI=60*15, inh_strength=-26, exc_strength=2)
grid50q = Setup(nrows=50, NE=50**2, NI=25**2, inh_strength=-22, exc_strength=4)
# grid50I = Setup(nrows=50, NE=0, NI=50**2, inh_strength=-10, exc_strength=3)
grid05 = Setup(nrows=5, NE=20, NI=5, inh_strength=-22, exc_strength=3)
current_setup = grid60q

# toroid
SPACE_WIDTH = current_setup.nrows
SPACE_HEIGHT = SPACE_WIDTH

# NEURONS
# Typically in the neocortex the ratio of Ex and Inh neurons is 80-20
NE = current_setup.nrows ** 2
NI = (current_setup.nrows // 2) ** 2
N = NE + NI
INH_STRENGTH = current_setup.inh_strength
EXH_STRENGTH = current_setup.exc_strength
# time constants of the neurons (in ms)
TAU = 10


# miscellaneous
DEF_VALUE = -1
POPULATION_FILENAME = "Population_{}.bn"
