#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-04-22

@author: Hauke Wernecke
"""

import numpy as np
from collections import OrderedDict

from .baseconfig import BaseConfig
from class_lib import Landscape, ExternalDrive, Synapse, TransferFunction

class BrianConfig(BaseConfig):
    WARMUP = 250 ###############################
    sim_time = 1200
    rows = 70
    # WARMUP = 500 ###############################
    # sim_time = 2500
    # rows = 70

    ##################### Patches
    center_range = OrderedDict({
        # "repeater": (8, 34),
        # "repeater": (17, 34),
        # "starter": (43, 68),
        # "linker": (16, 56),
    })

    RADIUSES = 6,
    AMOUNT_NEURONS = 50,
    PERCENTAGES = .2, -.2

    transfer_function = TransferFunction(50., .25)
    drive = ExternalDrive(25., 20., seeds=np.arange(2))
    synapse = Synapse(weight=.75, EI_factor=6.)
    landscape = Landscape("Perlin_uniform", stdE=3., stdI=2.25, connection_probability=.175, shift=1., params={"size": 4, "base": 1}, seed=0)
    # Induced spots of sustained activity: base 2
    # Not in base 3,

    transfer_function = TransferFunction(50., .3)
    drive = ExternalDrive(20., 20., seeds=np.arange(2))
    synapse = Synapse(weight=.75, EI_factor=6.)
    landscape = Landscape("Perlin_uniform", stdE=3., stdI=2.25, connection_probability=.175, shift=1., params={"size": 4, "base": 4}, seed=0)
