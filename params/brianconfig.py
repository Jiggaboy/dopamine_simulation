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
    # WARMUP = 500 ###############################
    sim_time = 2500
    rows = 80

    ##################### Patches
    center_range_70 = OrderedDict({
        # "in": (56, 48),
        # "repeater": (7, 38),
        # # "out-activator": (57, 29),
        # "starter": (49, 4),
    })
    center_range = OrderedDict({
        # "in": (56, 48),
        "repeater": (9, 41),
        "out-activator": (64, 32),
        # "starter": (49, 4),
    })

    RADIUSES = 8,
    AMOUNT_NEURONS = 50,
    PERCENTAGES = .2, -.2

    transfer_function = TransferFunction(50., .25)
    drive = ExternalDrive(20., 20., seeds=np.arange(2))
    synapse = Synapse(weight=.75, EI_factor=6.)
    landscape = Landscape("Perlin_uniform", stdE=3., stdI=2.25, connection_probability=.175, shift=1., params={"size": 4, "base": 1}, seed=0)
    # Induced spots of sustained activity: base 2
    # Not in base 3,

    landscape = Landscape("Perlin_uniform", stdE=3., stdI=2.25, connection_probability=.175, shift=1., params={"size": 4, "base": 1}, seed=0)

    drive = ExternalDrive(20., 20., seeds=np.arange(2))
    synapse = Synapse(weight=1., EI_factor=7.)
    landscape = Landscape("Perlin_uniform", stdE=3.25, stdI=2.5, connection_probability=.125, shift=1., params={"size": 4, "base": 1}, seed=0)


class GateConfig(BrianConfig):
    # AMOUNT_NEURONS = 75,
    center_range = OrderedDict({
        # "link": (71, 71),
        # "link-double": ((71, 71), (61, 77)),
        "gate-low-left": (29, 17),
        "gate-low-right": (44, 12),
        "gate-top-left": (33, 43),
        "gate-top-right": (31, 56),
    })

    def __post_init__(self):
        print("Run post_init in subclass...")
        self.landscape.params["base"] = 3
        super().__post_init__()


class SelectConfig(BrianConfig):

    center_range = OrderedDict({
        # BASE 4
        # "select1": (18, 66), # Rather a starter
        # "select2": (50, 65), # Affected by another branch
        # BASE 5
        # "link": (66, 63),
        # "repeat": (43, 11),
        # "gate": (56, 2),
        # "gate-left": (16, 58),
        # BASE 6
        # "select": (43, 29),
        # BASE 8
        # "activator": (75, 66),
    })

    def __post_init__(self):
        # self.RADIUSES = 8,
        # self.PERCENTAGES = .2,
        print("Run post_init in subclass...")
        self.landscape.params["base"] = 9

        super().__post_init__()
