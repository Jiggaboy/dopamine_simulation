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

class BrianConfig(BaseConfig):
    WARMUP = 500
    sim_time = 2500
    rows = 80

    RADIUSES = 6,
    AMOUNT_NEURONS = 50,
    PERCENTAGES = -.2, .1, .2
    # PERCENTAGES = .1, .2
    # PERCENTAGES = .2,

    RADIUSES = 6, #8,
    # AMOUNT_NEURONS = 50,
    # PERCENTAGES = .2, -.2

    transfer_function = TransferFunction(50., .25)

    drive = ExternalDrive(20., 20., seeds=np.arange(2))
    # drive = ExternalDrive(20., 20., seeds=np.arange(4))
    # drive = ExternalDrive(10., 30., seeds=np.arange(2))
    synapse = Synapse(weight=1., EI_factor=6.5)
    landscape = Landscape("Perlin_uniform", stdE=3.25, stdI=2.25, connection_probability=.125, shift=1.,
                          params={"size": 4, "base": 1}, seed=0)


class RandomLocationConfig(BrianConfig):
    base = 200
    n_locations = 20

    def __post_init__(self):
        super().__post_init__()
        generator = np.random.default_rng(seed=0)
        locations = generator.integers(0, self.rows, size=(self.n_locations, 2)).T
        self.center_range = OrderedDict({f"loc-{i}": locations[:, i] for i in range(self.n_locations)})
        self.center_range["repeater"] = (31, 18)
        self.center_range["activater"] = (5, 29)
        self.center_range["starter"] = (54, 57)


    def _add_detection_spots(self) -> None:
        detection_spots = []

        # SIZE 4, BASE 200
        loc = ((10, 10), (20, 10)) # pre, right, activated
        for i in range(20):
            UNI.append_spot(detection_spots, f"loc-{i}", loc)
        UNI.append_spot(detection_spots, "repeater", loc)
        UNI.append_spot(detection_spots, "activater", loc)
        UNI.append_spot(detection_spots, "starter", loc)
        return detection_spots


class GateConfig(BrianConfig):
    base = 56
    transfer_function = TransferFunction(50., .5)
    drive = ExternalDrive(10., 40., seeds=np.arange(1))
    drive = ExternalDrive(10., 40., seeds=np.arange(5))
    synapse = Synapse(weight=.9, EI_factor=7.75)
    synapse = Synapse(weight=.45, EI_factor=7.75)
    synapse = Synapse(weight=.225, EI_factor=7.75)
    synapse = Synapse(weight=.23, EI_factor=7.75)
    synapse = Synapse(weight=.3, EI_factor=7.75)
    # 56 - Edges sharp, but parameter tweaking could help
    landscape = Landscape("Perlin_uniform", stdE=3.25, stdI=3.35,
                            # connection_probability=.125,
                            # connection_probability=.25,
                            # connection_probability=.5,
                            # connection_probability=.5,
                            connection_probability=.385,
                          shift=1., params={"size": 4, "base": 56}, seed=0)
    center_range = OrderedDict({
        "gate-left": (30, 16),
        "gate-right": (36, 38),
    })

    PERCENTAGES = -.1, -.2,
    RADIUSES = 8,

    def _add_detection_spots(self) -> None:
        detection_spots = []

        center_gate = ((30, 17), (36, 35), (19, 37), ) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center_gate)
        UNI.append_spot(detection_spots, "gate-right", center_gate)
        return detection_spots


class GateRepeatConfig(BrianConfig):
    base = 5
    center_range = OrderedDict({
        "repeat": (3, 50),
        "repeat-early": (10, 52),
        # # "gate": (56, 2),
        # # "gate-left": (16, 58),
        # "starter": (58, 51),
    })


    def _add_detection_spots(self) -> None:
        detection_spots = []
        center_gate = ((17, 42), (20, 61), (1, 50)) #  left, right, merged
        # center_starter = (58, 60), (55, 73), (56, 2) # pre, post, center
        center_starter = (60, 52), (53, 73), (56, 66) # pre, post, center
        # center_repeater = (57, 8), (38, 28), (15, 42) # pre, post, reference
        center_repeater = (19, 60), (75, 62) # pre, post

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


    def _add_detection_spots(self) -> None:
        detection_spots = []
        center_activator = ((32, 59), (17, 3), (15, 50)) # pre, right, activated
        UNI.append_spot(detection_spots, "activator", center_activator)
        return detection_spots


class LinkerConfig(BrianConfig):
    base = 9
    center_range = OrderedDict({
        "link": (64, 52),
        # "link-left": (64, 52), # Remember to only activate left half and right half seperately.
        # "link-right": (64, 52),
    })
    PERCENTAGES = .2,


    def _add_detection_spots(self) -> None:
        detection_spots = []
        center_link = ((79, 62), (71, 27), (56, 63)) # main-pre, main-past, path-past
        UNI.append_spot(detection_spots, "link", center_link)
        return detection_spots



class SelectConfig(BrianConfig):
    base = 6

    center_range = OrderedDict({
        "gate-left": (55, 1),
        "gate-right": (53, 16),
        "select": (43, 29),
        "select-alt": (51, 13),
    })


    def _add_detection_spots(self) -> None:
        detection_spots = []
        center_select = ((55, 39), (53, 20), (37, 38)) # base, left, right
        UNI.append_spot(detection_spots, "select", center_select)
        UNI.append_spot(detection_spots, "select-alt", center_select)
        center_select = ((60, 0), (53, 23), (34, 9)) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center_select)
        UNI.append_spot(detection_spots, "gate-right", center_select)
        return detection_spots
