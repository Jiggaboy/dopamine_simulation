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
    transfer_function = TransferFunction(50., .5)
    drive = ExternalDrive(10., 40., seeds=np.arange(1))
    drive = ExternalDrive(10., 40., seeds=np.arange(5))
    synapse = Synapse(weight=.45, EI_factor=7.75)
    synapse = Synapse(weight=.225, EI_factor=7.75)
    # 56 - Edges sharp, but parameter tweaking could help
    landscape = Landscape("Perlin_uniform", stdE=3.25, stdI=3.35, connection_probability=.5, shift=1.,
                              params={"size": 4, "base": 56}, seed=0)
    # 58
    # landscape = Landscape("Perlin_uniform", stdE=3.4, stdI=3.5, connection_probability=.5, shift=1.,
    #                           params={"size": 4, "base": 56}, seed=0)
    # landscape = Landscape("Perlin_uniform", stdE=3.4, stdI=3.5, connection_probability=.5, shift=1.,
    #                           params={"size": 4, "base": 59}, seed=0)

    WARMUP = 500.
    sim_time = 3000.


    center_range = OrderedDict({
        "gate-left": (30, 16),
        "gate-right": (36, 38),
    })
    PERCENTAGES = -.1, #-.2,
    RADIUSES = 8,

    def _add_detection_spots(self) -> None:
        detection_spots = []

        center_gate = ((30, 17), (36, 35), (19, 37), ) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center_gate)
        UNI.append_spot(detection_spots, "gate-right", center_gate)
        return detection_spots

### Set the current config for all scripts/analyses here:
config = ExploreConfig()
# config = RandomLocationConfig()
