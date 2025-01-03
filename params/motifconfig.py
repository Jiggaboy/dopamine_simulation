#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Here various Configartions are defined that give access to one or more of the following motifs:
        Start, Repeat, Stop, Gate, Select, FakeRepeat, Link
@author: Hauke Wernecke
"""
from cflogger import logger

import numpy as np
from collections import OrderedDict

from params.baseconfig import BaseConfig
from class_lib import Landscape, ExternalDrive, Synapse, TransferFunction
from lib import universal as UNI

class MotifConfig(BaseConfig):
    WARMUP = 400
    sim_time = 4000
    rows = 80

    PERCENTAGES = .15,
    radius = 6,
    AMOUNT_NEURONS = 50,

    transfer_function = TransferFunction(50., .25)
    synapse = Synapse(weight=.3, EI_factor=8.)
    drive = ExternalDrive(10., 30., seeds=np.arange(2))

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 0, "octaves": 2, "persistence": .5,}, seed=0)


class SelectConfig(MotifConfig):
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 6, "octaves": 2, "persistence": .5,}, seed=0)

    PERCENTAGES = -.1, .1, .2,
    radius = 6, # 8,
    AMOUNT_NEURONS = 50,

    center_range = OrderedDict({
        "select-left": (26, 21),
        "select-right": (19, 13),
    })


    def _add_detection_spots(self) -> list:
        detection_spots = []
        center = ((11, 26), (31, 25), (8, 10)) # base, left, right
        UNI.append_spot(detection_spots, "select-left", center)
        UNI.append_spot(detection_spots, "select-right", center)
        return detection_spots


class GateConfig(MotifConfig):
    PERCENTAGES = -.2, .1
    radius = 6,
    AMOUNT_NEURONS = 50,

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 9, "octaves": 2, "persistence": .5,}, seed=0)

    center_range = OrderedDict({
        "gate-left": (42, 69),
        "gate-right": (29, 70),
    })

    def _add_detection_spots(self) -> list:
        detection_spots = []
        center_gate= ((46, 79), (28, 75), (28, 52)) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center_gate)
        UNI.append_spot(detection_spots, "gate-right", center_gate)
        return detection_spots


class RepeatConfig(MotifConfig):
    drive = ExternalDrive(5., 30., seeds=np.arange(4))
    # drive = ExternalDrive(5., 30., seeds=np.arange(2))
    PERCENTAGES = -.2, -.1, .1, .2,
    PERCENTAGES = -.2, .2,
    radius = 6,

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.45, "base": 10, "octaves": 2, "persistence": .5,}, seed=0)

    center_range = OrderedDict({
        # "repeat": (7, 2),
        # "repeat-alt": (7, 2), # difference in detection spots to repeat.
        "repeat-main": (40, 64), # A repeater patch on the main branch
    })


    def _add_detection_spots(self) -> list:
        detection_spots = []
        UNI.append_spot(detection_spots, "repeat", ((10, 11), (9, 69))) # early: (11, 31), , pre, post
        UNI.append_spot(detection_spots, "repeat-alt", ((10, 11), (9, 69), (3, 65))) # pre, post, right
        UNI.append_spot(detection_spots, "repeat-main", ((45, 71), (38, 54))) # pre, post
        return detection_spots


class FakeRepeatConfig(RepeatConfig):
    PERCENTAGES = .2,
    center_range = OrderedDict({
        "fake-repeat": (37, 57), # later than the main-repeat, establishes a starter in the second half of the branch.
        "anti-repeat": (39, 54), # later than the main-repeat, establishes a starter in the second half of the branch.
    })

    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(6) # Only updating the number, not the values of mean and std.


    def _add_detection_spots(self) -> list:
        detection_spots = []
        UNI.append_spot(detection_spots, "fake-repeat", ((45, 71), (38, 54))) # pre, post
        UNI.append_spot(detection_spots, "anti-repeat", ((45, 71), (38, 54))) # pre, post
        return detection_spots


class StartConfig(RepeatConfig):
    # PERCENTAGES = .1, .2,
    PERCENTAGES = .2,

    center_range = OrderedDict({
        "start": (45, 12),
    })


    def _add_detection_spots(self) -> list:
        detection_spots = []
        UNI.append_spot(detection_spots, "start", ((44, 8), (46, 76), )) # pre, post
        return detection_spots


class RandomLocationConfig(RepeatConfig):
    PERCENTAGES = .2, -.2
    n_locations = 32
    radius = 6,
    # radius = 80

    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(6) # Only updating the number, not the values of mean and std.
        generator = np.random.default_rng(seed=0)
        locations = generator.integers(0, self.rows, size=(self.n_locations, 2)).T # 1st location remains the same even for more locations with this style.
        self.center_range = OrderedDict({f"loc-{i}": locations[:, i] for i in range(self.n_locations)})
        logger.info("Center")
        for name, loc in self.center_range.items():
            logger.info(f"{name}: {loc}")


    def _add_detection_spots(self) -> list:
        detection_spots = []
        loc = ((10, 10), (20, 10))
        for i in range(self.n_locations):
            UNI.append_spot(detection_spots, f"loc-{i}", loc)
        return detection_spots


class LinkConfig(MotifConfig):
    base = 22
    drive = ExternalDrive(5., 30., seeds=np.arange(4))

    PERCENTAGES = .1,
    radius = 8,
    AMOUNT_NEURONS = 30, 50

    center_range = OrderedDict({
        "link": (40, 40),
    })
