#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:46:26 2022

@author: hauke
"""

from collections import OrderedDict

from .perlinconfig import PerlinConfig
from custom_class import Landscape, Synapse, TransferFunction

class NullConfig(PerlinConfig):
    WARMUP = 500
    sim_time = 5000.

    ##################### Patches
    center_range = OrderedDict({
        "null": (0.5, 0.5),
    })
    
    RADIUSES = .1,
    AMOUNT_NEURONS = 0,
    PERCENTAGES = .2,
    
    landscape = Landscape("Perlin_uniform", stdE=5., stdI=5., connection_probability=.2, shift=1., params={"size": 4, "base": 1}, seed=0)
