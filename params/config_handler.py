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
from params.motifconfig import GateConfig, SelectConfig

from collections import OrderedDict
from class_lib import ExternalDrive
import numpy as np
import lib.universal as UNI
from class_lib import Landscape, Synapse, ExternalDrive, TransferFunction
class ExploreConfig(BrianConfig):
    # rows = 60
    transfer_function = TransferFunction(50., .25)
    drive = ExternalDrive(5., 30., seeds=np.arange(5))
    drive = ExternalDrive(5., 30., seeds=np.arange(1))
    synapse = Synapse(weight=.3, EI_factor=7.5)

    transfer_function = TransferFunction(50., .5)
    synapse = Synapse(weight=.3, EI_factor=8.)
    drive = ExternalDrive(5., 30., seeds=np.arange(1))
    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.5, "base": 6}, seed=0)
    # Base 16: potential linker
    # Starter: size 2, base 4
    # Gate: {"size": 2., "base": 1}
    # Gate: {"size": 2.5, "base": 6}

    # ## Perlin noise
    # landscape = Landscape("Perlin_uniform", stdE=3.25, stdI=3.35, shift=1.,
    #                         # connection_probability=.325,
    #                         connection_probability=.38,
    #                         params={"size": 4, "base": 2}, seed=0)

    WARMUP = 250.
    sim_time = 2750.


    PERCENTAGES = .2,
    RADIUSES = 6,
    AMOUNT_NEURONS = 50,

    center_range = OrderedDict({
        "select-left": (26, 21),
        "select-right": (14, 12),
        # "gate-left": (39, 38),
        # "gate-right": (40, 25),
    })


    def _add_detection_spots(self) -> None:
        detection_spots = []
        center = ((11, 32), (28, 24), (7, 8)) # base, left, right
        UNI.append_spot(detection_spots, "select-left", center)
        UNI.append_spot(detection_spots, "select-right", center)
        center_gate= ((36, 42), (31, 25), (51, 30)) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center_gate)
        UNI.append_spot(detection_spots, "gate-right", center_gate)
        return detection_spots

### Set the current config for all scripts/analyses here:
config = ExploreConfig()
config = SelectConfig()
