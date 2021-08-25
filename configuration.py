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
sim_time = 15000. # ms
sim_time = 1000. # ms
# sim_time = 1. # warmup only


# Parameter space
center_range = {
    "repeater": (17, 34), # repeater
    "starter": (43, 68), # starter
    "linker": (16, 56), # linker
    "in-activator": (66, 34), # in-activator
    "edge-activator": (63, 34), # edge-activator
    "out-activator": (59, 34), # out-activator
    "in": (35, 18), # in
    "edge": (35, 22), # edge
    "out": (35, 26), # out
}

RADIUSES = (6, 12, 18)
AMOUNT_NEURONS = (10, 50, 100)
PERCENTAGES = (.3, .2, .1)
P_synapses = (1., .8, .6)



# Population
CAP = 2
Setup = namedtuple("Setup", ("nrows", "J", "g", "ext_mean", "ext_std", "tf_steepness", "tf_offset"))

# 70x70
grid_40_44r = Setup(nrows=70, J=2, g=7., ext_mean=25.0, ext_std=20.0, tf_steepness=.5, tf_offset=50)
grid_41_55 = Setup(nrows=70, J=2, g=6.5, ext_mean=20.0, ext_std=20.0, tf_steepness=.5, tf_offset=50) # Good setup, approved by Andrew


# grid_33_45_4 = Setup(nrows=60, J=3., g=8., ext_mean=10.0, ext_std=30.0) #Perlin 3
# 5x5
# grid05 = Setup(nrows=5, J=5.5, g=3.0, ext_mean=0.5, ext_std=4.0)


current_setup = grid_41_55

# Transfer function (sigmoid)
STEEPNESS = current_setup.tf_steepness
X_OFFSET = current_setup.tf_offset

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
TAU = 12
# Update synapses
ETA = .1


# miscellaneous
DEF_VALUE = -1
PERLIN_SIZE = 4
POPULATION_FILENAME = "Population_{}_{}.bn"
