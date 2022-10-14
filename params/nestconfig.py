#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-10-13

@author: Hauke Wernecke
"""

import numpy as np
from collections import namedtuple, OrderedDict

from .baseconfig import BaseConfig
from custom_class import Landscape, ExternalDrive, Synapse, TransferFunction

class NestConfig(BaseConfig):
    WARMUP = 500 ###############################
    sim_time = 5000
    rows = 40

    synapse = Synapse(weight=.75, EI_factor=6.5)
    transfer_function = TransferFunction(50., .25)
    drive = ExternalDrive(15., 20., seeds=np.arange(2))
    landscape = Landscape("Perlin_uniform", stdE=5., stdI=5., connection_probability=.2, shift=1., params={"size": 4, "base": 1}, seed=0)
