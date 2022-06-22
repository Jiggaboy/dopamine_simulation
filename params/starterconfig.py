#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:46:26 2022

@author: hauke
"""

from collections import namedtuple, OrderedDict
from .perlinconfig import PerlinConfig
from custom_class import Landscape, Synapse, TransferFunction

class StarterConfig(PerlinConfig):
    # WARMUP = 10
    sim_time = 5000.

    center_range = OrderedDict({
        "starter-CL": (19, 24),
        "starter-TR": (55, 56),
        "starter-TC": (45, 61),
        # "starter4": (42, 69),
        # "starter5": (44, 69),
    })

    landscape = Landscape("Perlin_uniform", stdE=5., stdI=5., connection_probability=.2, shift=1., params={"size": 4, "base": 10}, seed=0)
