#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'


__all__ = [
    "config"
]
#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

from params import TestConfig
from params import BaseConfig
from params import BrianConfig, GateConfig, SelectConfig, GateRepeatConfig, RandomLocationConfig, LinkerConfig


from collections import OrderedDict
from class_lib import ExternalDrive
import numpy as np
import lib.universal as UNI
from class_lib import Landscape, Synapse, ExternalDrive, TransferFunction
class ExploreConfig(BrianConfig):
    # rows = 60
    transfer_function = TransferFunction(50., .5)
    drive = ExternalDrive(15., 40., seeds=np.arange(5))
    # synapse = Synapse(weight=.35, EI_factor=7.7)
    synapse = Synapse(weight=.3, EI_factor=8.)
    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=2.8, stdI=3., shift=1.,
                            # connection_probability=.325,
                            connection_probability=.375,
                            params={"size": 2.45, "base": 6}, seed=0)
    # ## Perlin noise
    # landscape = Landscape("Perlin_uniform", stdE=3.25, stdI=3.35, shift=1.,
    #                         # connection_probability=.325,
    #                         connection_probability=.38,
    #                         params={"size": 4, "base": 2}, seed=0)

    WARMUP = 250.
    sim_time = 2000.


    center_range = OrderedDict({
        ### Base 62
        # "select-right": (49, 58),
        # "select-left": (39, 60),
        ### Base 6
        "select-left": (24, 21),
        "select-right": (19, 15),
    })
    PERCENTAGES = .15,
    RADIUSES = 6,
    AMOUNT_NEURONS = 30, 50,


    def _add_detection_spots(self) -> None:
        detection_spots = []

        ### Base 62
        # center = ((39, 47), (29, 66), (57, 67), ) # base, left, right
        ### Base
        center = ((11, 32), (28, 24), (7, 8)) # base, left, right
        UNI.append_spot(detection_spots, "select-left", center)
        UNI.append_spot(detection_spots, "select-right", center)

        return detection_spots

### Set the current config for all scripts/analyses here:
config = ExploreConfig()
# config = GateConfig()
