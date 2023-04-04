#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-04-22

@author: Hauke Wernecke
"""

import numpy as np
from collections import OrderedDict

from .baseconfig import BaseConfig
from custom_class import Landscape, ExternalDrive, Synapse, TransferFunction

class PerlinConfig(BaseConfig):
    WARMUP = 200 ###############################
    sim_time = 1500
    rows = 70
    # WARMUP = 500 ###############################
    # sim_time = 2500
    # rows = 70

    ##################### Patches
    center_range = OrderedDict({
        # "repeater": (17, 34),
        # "starter": (43, 68),
        # "linker": (16, 56),
        # # #"in-activator": (66, 34),
        # # #"edge-activator": (63, 34),
        # "out-activator": (59, 34),
        # "in": (35, 18),
        # # #"edge": (35, 22),
        # # #"out": (35, 26),
        # "gate-top": (29, 34),
        # "gate-bottom": (28, 26),
    })

    RADIUSES = 6,
    AMOUNT_NEURONS = 50,
    PERCENTAGES = .2, -.2

    synapse = Synapse(weight=.75, EI_factor=6.5)
    transfer_function = TransferFunction(50., .25)
    drive = ExternalDrive(25., 20., seeds=np.arange(2))
    landscape = Landscape("Perlin_uniform", stdE=3.5, stdI=2., connection_probability=.2, shift=1., params={"size": 4, "base": 1}, seed=0)
    # landscape = Landscape("Perlin_uniform", stdE=5., stdI=5., connection_probability=.2, shift=1., params={"size": 4, "base": 1}, seed=0)
