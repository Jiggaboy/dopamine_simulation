#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:38:45 2022

@author: hauke
"""



import numpy as np
from collections import OrderedDict
from .baseconfig import BaseConfig

from custom_class import Landscape, ExternalDrive, Synapse, TransferFunction


class TestConfig(BaseConfig):
    ##################### Time
    sim_time = 500 # ms

    # Parameter space
    center_range = OrderedDict({
        "repeater": (17, 34), # repeater
    })

    RADIUSES = (12, )
    AMOUNT_NEURONS = (50, )
    PERCENTAGES = (.1, )
    P_synapses = (1., )

    rows = 36

    landscape = Landscape("Perlin_uniform", stdE=3., stdI=2., connection_probability=1., shift=1., params={"size": 3, "base": 3}, seed=None)
    synapse = Synapse(weight=.5, EI_factor=6)
    transfer_function = TransferFunction(50., .15)
    drive = ExternalDrive(20., 20., seeds=(0,))

    # landscape = Landscape("Perlin_uniform", params={"size": 4, "stdE": 4., "stdI": 4.})
    # landscape = Landscape("symmetric", params={"size": 2, "stdE": 3., "stdI": 2.})
    # landscape = Landscape("homogeneous", params={"phi": 2, "stdE": 3., "stdI": 2.})
    # landscape = Landscape("random", params={"stdE": 3., "stdI": 2.})
