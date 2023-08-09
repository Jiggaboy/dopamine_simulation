#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:46:26 2022

@author: hauke
"""

from collections import namedtuple, OrderedDict

from .baseconfig import BaseConfig
from class_lib import Landscape, Synapse, TransferFunction

class ScaleupConfig(BaseConfig):
    WARMUP = 500
    sim_time = 5000.
    rows = 80

    ##################### Patches
    center_range = OrderedDict({
        #"repeater": (17, 34),
        #"starter": (43, 78),
        #"linker": (16, 56),
        #"in-activator": (66, 34),
        #"edge-activator": (63, 34),
        #"out-activator": (61, 34),
        #"in": (35, 18),
        #"edge": (35, 22),
        #"out": (35, 26),
    })

    RADIUSES = 6,
    AMOUNT_NEURONS = 50,
    PERCENTAGES = .2,

    synapse = Synapse(weight=.65, EI_factor=7)
    transfer_function = TransferFunction(50., .25)
    landscape = Landscape("Perlin_uniform", stdE=5., stdI=5., connection_probability=.2, shift=1., params={"size": 4, "base": 1}, seed=0)
