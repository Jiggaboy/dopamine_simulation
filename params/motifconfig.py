#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-04-22

@author: Hauke Wernecke
"""
from cflogger import logger

import numpy as np
from collections import OrderedDict

from params.baseconfig import BaseConfig
from class_lib import Landscape, ExternalDrive, Synapse, TransferFunction
from lib import universal as UNI

class MotifConfig(BaseConfig):
    WARMUP = 500
    sim_time = 2500
    rows = 80

    transfer_function = TransferFunction(50., .5)
    synapse = Synapse(weight=.3, EI_factor=8.)
    drive = ExternalDrive(10., 40., seeds=np.arange(5))


## Gate: 2.45 > 2.4
## Select: 2.5 ~ 2.45 > 2.4
class SelectConfig(MotifConfig):
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.4, "base": 6}, seed=0)

    PERCENTAGES = .2,
    RADIUSES = 6,
    AMOUNT_NEURONS = 50,

    center_range = OrderedDict({
        "select-left": (24, 21),
        "select-right": (15, 16),
    })


    def _add_detection_spots(self) -> None:
        detection_spots = []
        center = ((11, 32), (28, 24), (7, 8)) # base, left, right
        UNI.append_spot(detection_spots, "select-left", center)
        UNI.append_spot(detection_spots, "select-right", center)
        return detection_spots


class GateConfig(MotifConfig):
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.45, "base": 6}, seed=0)

    PERCENTAGES = -.2,
    RADIUSES = 6,
    AMOUNT_NEURONS = 50,

    center_range = OrderedDict({
        "gate-left": (39, 35),
        "gate-right": (40, 26),
    })


    def _add_detection_spots(self) -> None:
        detection_spots = []
        center_gate= ((36, 42), (31, 25), (51, 30)) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center_gate)
        UNI.append_spot(detection_spots, "gate-right", center_gate)
        return detection_spots


class GateConfig_small(MotifConfig):
    transfer_function = TransferFunction(50., .25)
    drive = ExternalDrive(5., 30., seeds=np.arange(1))
    synapse = Synapse(weight=.3, EI_factor=8.)
    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=2.4, stdI=2.8, shift=1.,
                            connection_probability=.3,
                            params={"size": 2., "base": 0}, seed=0)
    PERCENTAGES = -.2,
    RADIUSES = 6,
    AMOUNT_NEURONS = 50,

    center_range = OrderedDict({
        "gate-left": (39, 3),
        "gate-right": (50, 10),
    })


    def _add_detection_spots(self) -> None:
        detection_spots = []
        center = ((28, 4), (42, 16), (51, 75)) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center)
        UNI.append_spot(detection_spots, "gate-right", center)
        return detection_spots


class RepeatConfig(MotifConfig):
    transfer_function = TransferFunction(50., .25)
    drive = ExternalDrive(5., 30., seeds=np.arange(1))
    synapse = Synapse(weight=.3, EI_factor=8.)
    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=2.4, stdI=2.8, shift=1.,
                            connection_probability=.3,
                            params={"size": 2., "base": 11}, seed=0)
    PERCENTAGES = .2,
    RADIUSES = 6, # maybe even 8 or less increase, yet strong starter
    AMOUNT_NEURONS = 50,

    center_range = OrderedDict({
        "repeat": (60, 5),
    })


    def _add_detection_spots(self) -> None:
        detection_spots = []
        UNI.append_spot(detection_spots, "repeat", ((57, 7), (70, 1)))
        return detection_spots
