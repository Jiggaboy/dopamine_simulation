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
    sim_time = 1000. # ms

    # Parameter space
    center_range = OrderedDict({
        "repeater": (17, 34), # repeater
    })

    RADIUSES = (12, )
    AMOUNT_NEURONS = (50, )
    PERCENTAGES = (.1, )
    P_synapses = (1., )

    rows = 40

    landscape = Landscape("Perlin_uniform", stdE=2., stdI=3., connection_probability=1., shift=.5, params={"size": 2, "base": 3}, seed=None)
    synapse = Synapse(weight=.25, EI_factor=1.25)
    transfer_function = TransferFunction(50., .15)

    drive = ExternalDrive(20., 30.)
