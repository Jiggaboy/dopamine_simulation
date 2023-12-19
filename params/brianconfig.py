#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-04-22

@author: Hauke Wernecke
"""
from cflogger import logger

import numpy as np
from collections import OrderedDict

from .baseconfig import BaseConfig
from class_lib import Landscape, ExternalDrive, Synapse, TransferFunction

class BrianConfig(BaseConfig):
    def __post_init__(self):
        if hasattr(self, "base"):
            logger.info("Set new base for landscape.")
            self.landscape.params["base"] = self.base

        super().__post_init__()


    WARMUP = 500
    sim_time = 2500
    ###############################
    # WARMUP = 250
    # sim_time = 1200
    ###############################
    rows = 80

    ##################### Patches
    center_range = OrderedDict({})

    RADIUSES = 6,
    AMOUNT_NEURONS = 50,
    PERCENTAGES = .1, .2

    RADIUSES = 6, 8,
    # AMOUNT_NEURONS = 50,
    # PERCENTAGES = .2, -.2

    transfer_function = TransferFunction(50., .25)
    drive = ExternalDrive(20., 20., seeds=np.arange(2))
    synapse = Synapse(weight=.75, EI_factor=6.)
    landscape = Landscape("Perlin_uniform", stdE=3., stdI=2.25, connection_probability=.175, shift=1.,
                          params={"size": 4, "base": 1}, seed=0)
    # Induced spots of sustained activity: base 2. Not in base 3.

    drive = ExternalDrive(20., 20., seeds=np.arange(2))
    drive = ExternalDrive(20., 20., seeds=np.arange(4))
    # drive = ExternalDrive(10., 30., seeds=np.arange(3))
    synapse = Synapse(weight=1., EI_factor=7.)
    landscape = Landscape("Perlin_uniform", stdE=3.25, stdI=2.5, connection_probability=.125, shift=1.,
                          params={"size": 4, "base": 1}, seed=0)


class RandomLocationConfig(BrianConfig):
    base = 200
    n_locations = 20

    def __post_init__(self):
        super().__post_init__()
        # generator = np.random.default_rng(seed=0)
        # locations = generator.integers(0, self.rows, size=(self.n_locations, 2)).T
        # self.center_range = OrderedDict({f"loc-{i}": locations[:, i] for i in range(self.n_locations)})
        self.center_range["repeater"] = (31, 18)
        # self.center_range["activater"] = (5, 29)
        # self.center_range["starter"] = (54, 57)
        print(self.center_range)


class GateConfig(BrianConfig):
    base = 3
    center_range = OrderedDict({
        # "gate-low-left": (29, 17),
        # "gate-low-right": (44, 12),
        "gate-top-left": (33, 43),
        "gate-top-right": (31, 56),
    })


class GateRepeatConfig(BrianConfig):
    base = 5
    center_range = OrderedDict({
        "repeat": (3, 50),
        "repeat-early": (10, 52),
        # "repeat": (43, 11), # What is this?
        # "gate": (56, 2),
        "gate-left": (16, 58),
        # "starter": (58, 51),
    })


    def _add_detection_spots(self) -> None:

        from lib import universal as UNI
        detection_spots = []

        center_gate = ((17, 42), (20, 61), (1, 50)) #  left, right, merged
        # center_starter = (58, 60), (55, 73), (56, 2) # pre, post, center
        center_starter = (60, 52), (53, 73), (56, 66) # pre, post, center
        # center_repeater = (57, 8), (38, 28), (15, 42) # pre, post, reference
        center_repeater = (19, 60), (75, 62) # left, pre, post

        UNI.append_spot(detection_spots, "gate-left", center_gate)
        UNI.append_spot(detection_spots, "starter", center_starter)
        UNI.append_spot(detection_spots, "repeat", center_repeater)
        UNI.append_spot(detection_spots, "repeat-early", center_repeater)
        return detection_spots




class ActivatorConfig(BrianConfig):
    base = 22
    center_range = OrderedDict({
        "activator": (17, 63),
    })


class LinkerConfig(BrianConfig):
    base = 9
    center_range = OrderedDict({
        "link": (64, 52),
        "link-left": (64, 52),
        "link-right": (64, 52),
    })



class SelectConfig(BrianConfig):
    base = 6

    center_range = OrderedDict({
        "select": (43, 29),
        "select-alt": (51, 13),
    })
