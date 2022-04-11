#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:38:45 2022

@author: hauke
"""



import numpy as np
from collections import OrderedDict
from .baseconfig import BaseConfig

from custom_class import Landscape


class TestConfig(BaseConfig):
    ##################### Time
    sim_time = 1000. # ms

    # Parameter space
    center_range = OrderedDict({
        "repeater": (17, 34), # repeater
    })

    RADIUSES = (12, )
    AMOUNT_NEURONS = (50, )
    PERCENTAGES = (.1, )
    P_synapses = (1., )

    rows = 28

    landscape = Landscape("Perlin_uniform", stdE=4., stdI=4., connection_probability=.2, shift=1., params={"size": 4}, seed=0)

    # landscape = Landscape("Perlin_uniform", params={"size": 4, "stdE": 4., "stdI": 4.})
    # landscape = Landscape("symmetric", params={"size": 2, "stdE": 3., "stdI": 2.})
    # landscape = Landscape("homogeneous", params={"phi": 2, "stdE": 3., "stdI": 2.})
    # landscape = Landscape("random", params={"stdE": 3., "stdI": 2.})
