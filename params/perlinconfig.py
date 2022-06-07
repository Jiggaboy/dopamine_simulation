#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:46:26 2022

@author: hauke
"""

from .baseconfig import BaseConfig
from custom_class import Landscape, Synapse, TransferFunction

class PerlinConfig(BaseConfig):
    # WARMUP = 10
    sim_time = 500.

    RADIUSES = 6,
    AMOUNT_NEURONS = 50,
    PERCENTAGES = .2,
    P_synapses = 1.,

    synapse = Synapse(weight=.5, EI_factor=7.)
    transfer_function = TransferFunction(50., .25)
    landscape = Landscape("Perlin_uniform", stdE=5., stdI=5., connection_probability=.3, shift=1., params={"size": 4, "base": 1}, seed=0)

    synapse = Synapse(weight=1., EI_factor=6.5)
    transfer_function = TransferFunction(50., .5)
    landscape = Landscape("Perlin_uniform", stdE=5., stdI=5., connection_probability=.3, shift=1., params={"size": 4, "base": 1}, seed=0)
