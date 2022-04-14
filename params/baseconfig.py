#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:38:45 2022

@author: hauke
"""


import logging
log = logging.getLogger()

import numpy as np
from collections import namedtuple, OrderedDict


from custom_class import ExternalDrive, Landscape, Plasticity, Synapse, TransferFunction


class BaseConfig:


    POPULATION_FILENAME = "Population_{}_{}.bn"
    PATH_CONNECTIVITY = "connectivity_matrix_{}_{}.bn"

    ##################### Time
    ###### In time steps [a.u.]
    WARMUP = 500
    sim_time = 15000.
    TAU = 12.


    ##################### Patches
    center_range = OrderedDict({
        "repeater": (17, 34),
        # "starter": (43, 68),
        # "linker": (16, 56),
        # "in-activator": (66, 34),
        # "edge-activator": (63, 34),
        # "out-activator": (59, 34),
        # "in": (35, 18),
        # "edge": (35, 22),
        # "out": (35, 26),
    })


    RADIUSES = (6, 12, 18)
    AMOUNT_NEURONS = (10, 50, 100)
    PERCENTAGES = (.3, .2, .1)
    P_synapses = (1., .8, .6)


    #####################
    rows = 70
    plasticity = Plasticity(rate=.1, cap=2.)
    synapse = Synapse(weight=2., EI_factor=6.5)
    transfer_function = TransferFunction(50., .5)
    drive = ExternalDrive(20., 20.)


    # landscape = Landscape("Perlin_uniform", params={"size": 4, "stdE": 5., "stdI": 5.})
    landscape = None


    def __init__(self):
        self.__post_init__()


    def __post_init__(self):
        log.info("\n".join(("Configuration:", f"Rows: {self.rows}", f"Landscape: {self.landscape}")))


    def path_to_connectivity_matrix(self):
        return self.PATH_CONNECTIVITY.format(self.landscape.mode, self.rows)
