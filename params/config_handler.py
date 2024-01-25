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
    # drive = ExternalDrive(10., 40., seeds=np.arange(5))
    drive = ExternalDrive(15., 40., seeds=np.arange(1))
    synapse = Synapse(weight=.3, EI_factor=8.)
    synapse = Synapse(weight=.3, EI_factor=8.)
    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=2.25, stdI=2.5, shift=1.,
                            connection_probability=.325,
                            # params={"size": 2.5, "base": 6}, seed=0)
                            params={"size": 1, "base": 6}, seed=0)

    # ## Perlin noise
    # landscape = Landscape("Perlin_uniform", stdE=3.25, stdI=3.35, shift=1.,
    #                         # connection_probability=.325,
    #                         connection_probability=.38,
    #                         params={"size": 4, "base": 2}, seed=0)

    WARMUP = 250.
    sim_time = 1500.


    center_range = OrderedDict({
        "select-left": (27, 20),
        "select-right": (18, 13),
    })
    PERCENTAGES = -.2,
    PERCENTAGES = .2,
    RADIUSES = 6,
    AMOUNT_NEURONS = 50,


    def _add_detection_spots(self) -> None:
        detection_spots = []

        # UNI.append_spot(detection_spots, "start", ((37, 71), (50, 75)))
        UNI.append_spot(detection_spots, "start", ((37, 71), (50, 75)))
        UNI.append_spot(detection_spots, "repeat", ((37, 71), (50, 75)))
        UNI.append_spot(detection_spots, "repeat-left", ((37, 71), (50, 75)))
        center = ((11, 32), (28, 24), (7, 8)) # base, left, right
        UNI.append_spot(detection_spots, "select-left", center)
        UNI.append_spot(detection_spots, "select-right", center)

        return detection_spots

### Set the current config for all scripts/analyses here:
config = ExploreConfig()
# config = GateConfig()
