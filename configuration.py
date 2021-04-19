#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:26:04 2021

@author: hauke
"""

from collections import namedtuple

# Time management
WARMUP = 50
# Duration of the simulation (in ms)
sim_time = 10800. # ms
sim_time = 1000. # ms
# sim_time = 1. # warmup only

# Transfer function (sigmoid)
STEEPNESS = .5
X_OFFSET = 14



# Population
CAP = 2
Setup = namedtuple("Setup", ("nrows", "J", "g", "ext_mean", "ext_std"))

# 60x60
grid_40_45 = Setup(nrows=60, J=3.0, g=6., ext_mean=2.0, ext_std=4.0)
grid_40_44 = Setup(nrows=60, J=2.5, g=6., ext_mean=0.0, ext_std=4.0) # consistent main with irregular branching
# grid_40_44r = Setup(nrows=60, J=2., g=6., ext_mean=5.0, ext_std=5.0)
grid_41_46r = Setup(nrows=60, J=1., g=6., ext_mean=2.0, ext_std=5.0) # repeater - unstable main
grid60sh = Setup(nrows=60, J=2.5, g=6., ext_mean=1.0, ext_std=4.0)
grid60h = Setup(nrows=60, J=2.5, g=6., ext_mean=0.0, ext_std=4.0)
grid60m = Setup(nrows=60, J=2.0, g=6., ext_mean=1.0, ext_std=4.0)
grid60l = Setup(nrows=60, J=1.6, g=6., ext_mean=2.0, ext_std=4.0)
grid60sl = Setup(nrows=60, J=2, g=8., ext_mean=0.5, ext_std=4.0)
# 5x5
grid05 = Setup(nrows=5, J=5.5, g=3.0, ext_mean=0.5, ext_std=4.0)


current_setup = grid_41_46r

# toroid
SPACE_WIDTH = current_setup.nrows
SPACE_HEIGHT = SPACE_WIDTH

# External drive
ext_input_mean = current_setup.ext_mean
ext_input_std = current_setup.ext_std

# NEURONS
# Typically in the neocortex the ratio of Ex and Inh neurons is 80-20
NE = current_setup.nrows ** 2
NI = (current_setup.nrows // 2) ** 2
N = NE + NI
J = current_setup.J
g = current_setup.g
INH_STRENGTH = - J * g
EXH_STRENGTH = J
# INH_STRENGTH = current_setup.inh_strength
# EXH_STRENGTH = current_setup.exc_strength
# time constants of the neurons (in ms)
TAU = 20
# Update synapses
ETA = .1


# miscellaneous
DEF_VALUE = -1
PERLIN_SIZE = 3
POPULATION_FILENAME = "Population_{}.bn"
