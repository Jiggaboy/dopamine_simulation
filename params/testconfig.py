#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:38:45 2022

@author: hauke
"""



import numpy as np
from collections import OrderedDict
from .baseconfig import BaseConfig

from class_lib import Landscape, ExternalDrive, Synapse, TransferFunction


class TestConfig(BaseConfig):
    ##################### Time
    sim_time = 1000 # ms

    # Parameter space
    center_range = OrderedDict({
    })

    rows = 40

    landscape = Landscape("Perlin_uniform", stdE=2., stdI=3.25, connection_probability=.125, shift=1.,
                          params={"size": 3, "base": 3}, seed=None)
    # landscape = Landscape("Perlin_uniform", stdE=3., stdI=2., connection_probability=1., shift=1., params={"size": 3, "base": 3}, seed=None)
    # landscape = Landscape("Perlin_uniform", stdE=3., stdI=2., connection_probability=1., shift=1., params={"size": 3, "base": 3}, seed=None)
    synapse = Synapse(weight=1., EI_factor=6)
    transfer_function = TransferFunction(50., .15)
    drive = ExternalDrive(15., 20., seeds=(0, 1))

    # synapse = Synapse(weight=1., EI_factor=7.)
    # landscape = Landscape("Perlin_uniform", stdE=3.25, stdI=2.5, connection_probability=.125, shift=1.,
    #                       params={"size": 4, "base": 1}, seed=0)

    # landscape = Landscape("Perlin_uniform", params={"size": 4, "stdE": 4., "stdI": 4.})
    # landscape = Landscape("symmetric", params={"size": 2, "stdE": 3., "stdI": 2.})
    # landscape = Landscape("homogeneous", params={"phi": 2, "stdE": 3., "stdI": 2.})
    # landscape = Landscape("random", params={"stdE": 3., "stdI": 2.})
