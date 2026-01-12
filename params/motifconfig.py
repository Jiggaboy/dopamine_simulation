#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Here various Configartions are defined that give access to one or more of the following motifs:
        Start, Repeat, Stop, Gate, Select, FakeRepeat, Link
@author: Hauke Wernecke
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'


#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

from cflogger import logger

import numpy as np
from collections import OrderedDict

from params.baseconfig import BaseConfig
from class_lib import Landscape, ExternalDrive, Synapse, TransferFunction
from lib import universal as UNI

class MotifConfig(BaseConfig):
    warmup = 400
    sim_time = 4000
    rows = 80

    PERCENTAGES = .15,
    radius = 6,
    AMOUNT_NEURONS = 50,

    transfer_function = TransferFunction(50., .25)
    synapse = Synapse(weight=.3, EI_factor=8.)
    drive = ExternalDrive(10., 30., seeds=np.arange(4))

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.5, "base": 0, "octaves": 2, "persistence": .5,}, seed=0)


class SelectConfig(MotifConfig):
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 6, "octaves": 2, "persistence": .5,}, seed=0)

    PERCENTAGES = -.2, .1, .2,
    radius = 6
    AMOUNT_NEURONS = 50,

    center_range = OrderedDict({
        "select-left": (26, 21),
        "select-right": (19, 13),
    })

    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(6)

    def _add_detection_spots(self) -> list:
        detection_spots = []
        center = ((11, 26), (31, 25), (8, 10)) # base, left, right
        UNI.append_spot(detection_spots, "select-left", center)
        UNI.append_spot(detection_spots, "select-right", center)
        return detection_spots


class GateConfig(MotifConfig):
    PERCENTAGES = .1, .2, -.2
    PERCENTAGES = .1, .2
    radius = 6,
    AMOUNT_NEURONS = 50,
    save_synaptic_input = True

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 9, "octaves": 2, "persistence": .5,}, seed=0)

    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(6)

    center_range = OrderedDict({
        "gate-left": (42, 69),
        "gate-right": (29, 70),
    })

    def _add_detection_spots(self) -> list:
        detection_spots = []
        # center_gate= ((46, 79), (28, 75), (28, 52)) # left, right, merged
        center_gate= ((45, 77), (28, 77), (28, 52)) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center_gate)
        UNI.append_spot(detection_spots, "gate-right", center_gate)
        return detection_spots


class RepeatConfig(MotifConfig):
    drive = ExternalDrive(5., 30., seeds=np.arange(6))
    # drive = ExternalDrive(5., 30., seeds=np.arange(2))
    PERCENTAGES = -.2, -.1, .1, .2,
    PERCENTAGES = -.2, .2,
    radius = 6,

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.45, "base": 10, "octaves": 2, "persistence": .5,},
                            seed=0)

    center_range = OrderedDict({
        # "repeat": (7, 2),
        # "repeat-alt": (7, 2), # difference in detection spots to repeat.
        "repeat-main": (40, 64), # A repeater patch on the main branch
    })


    def _add_detection_spots(self) -> list:
        detection_spots = []
        UNI.append_spot(detection_spots, "repeat", ((10, 11), (9, 69))) # early: (11, 31), , pre, post
        UNI.append_spot(detection_spots, "repeat-alt", ((10, 11), (9, 69), (3, 65))) # pre, post, right
        # UNI.append_spot(detection_spots, "repeat-main", ((45, 71), (38, 54))) # pre, post
        # UNI.append_spot(detection_spots, "repeat-main", ((45, 71), (35, 44))) # pre, post
        UNI.append_spot(detection_spots, "repeat-main", ((45, 71), (38, 45))) # pre, post
        UNI.append_spot(detection_spots, "con-repeat", ((45, 71), (38, 45))) # pre, post
        return detection_spots


class FakeRepeatConfig(RepeatConfig):
    PERCENTAGES = .2,
    center_range = OrderedDict({
        "fake-repeat": (37, 57), # later than the main-repeat, establishes a starter in the second half of the branch.
        "anti-repeat": (39, 54), # later than the main-repeat, establishes a starter in the second half of the branch.
        "con-repeat": (38, 55), # later than the main-repeat, establishes a starter in the second half of the branch.
    })

    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(6) # Only updating the number, not the values of mean and std.


    def _add_detection_spots(self) -> list:
        detection_spots = []
        # UNI.append_spot(detection_spots, "fake-repeat", ((45, 71), (35, 44))) # pre, post
        # UNI.append_spot(detection_spots, "anti-repeat", ((45, 71), (35, 44))) # pre, post
        # UNI.append_spot(detection_spots, "con-repeat", ((45, 71), (35, 44))) # pre, post
        UNI.append_spot(detection_spots, "fake-repeat", ((45, 71), (38, 45))) # pre, post
        UNI.append_spot(detection_spots, "anti-repeat", ((45, 71), (38, 45))) # pre, post
        UNI.append_spot(detection_spots, "con-repeat", ((45, 71), (38, 45))) # pre, post
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
    PERCENTAGES = .1, .2
    # PERCENTAGES = .1, -.1
    PERCENTAGES = .2, -.2
    n_locations = 32
    radius = 6
    # radius = 80
    AMOUNT_NEURONS = 40,

    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(2) #6# Only updating the number, not the values of mean and std.
        generator = np.random.default_rng(seed=0)
        locations = generator.integers(0, self.rows, size=(self.n_locations, 2)).T # 1st location remains the same even for more locations with this style.

        self.center_range = OrderedDict({
            "start": (45, 12),
            # "repeat": (7, 2),
            "repeat-main": (41, 66),
            # "repeat-main": (40, 64), # old -> 2 grid points later has less effect
            # "fake-repeat": (37, 57), # later than the main-repeat, establishes a starter in the second half of the branch.
            # "anti-repeat": (39, 54),
            "con-repeat": (38, 55),
            "low-1": (67, 38),
            "low-2": (55, 14),
            "low-3": (10, 51),
            "low-4": (29, 31),
            "low-5": (58, 75),
            "low-6": (38, 14),
            "high-1": (42, 25),
            "high-2": (53, 24),
            "high-3": (68, 28),
        })

        random_locations = OrderedDict({f"loc-{i}": tuple(locations[:, i]) for i in range(self.n_locations)})
        self.center_range.update(random_locations)
        # self.center_range = random_locations
        self.center_range.pop("loc-1", None) # Static bump if -20
        self.center_range.pop("loc-16", None) # Static bump with +20
        # self.center_range = {k: random_locations[k] for k in ('loc-19', )}

        # logger.info("Center")
        # _tmp = {}
        for name, loc in self.center_range.items():
            logger.info(f"{name}: {loc}")
        #     if name in ("loc-0", ):
        #         _tmp[name] = loc
        # self.center_range = _tmp

class LocationConfig(MotifConfig):
    drive = ExternalDrive(5., 30., seeds=np.arange(2))
    AMOUNT_NEURONS = 40,

    PERCENTAGES = .1, -.1
    PERCENTAGES = .2, -.2, .1, -.1, -.5, .5
    n_locations = 20
    radius = 6
    # radius = 80
    save_synaptic_input = False

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.5, "base": 300, "octaves": 2, "persistence": .75,},
                            seed=0)


    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(4) #6# Only updating the number, not the values of mean and std.
        generator = np.random.default_rng(seed=0)
        locations = generator.integers(0, self.rows, size=(self.n_locations, 2)).T # 1st location remains the same even for more locations with this style.

        self.center_range = OrderedDict({
            "repeat-1": (41, 68),
            "repeat-2": (50, 59),
            "high-1": (21, 26),
            "high-2": (48, 71),
            "gate-1": (15, 33),
            "gate-2": (0, 10),
            "start-1": (58, 10),
        })

        random_locations = OrderedDict({f"loc-{i}": tuple(locations[:, i]) for i in range(self.n_locations)})
        # self.center_range.update(random_locations)
        # self.center_range = random_locations
        # self.center_range.pop("loc-1", None) # Static bump if -20
        # self.center_range.pop("loc-16", None) # Static bump with +20
        # self.center_range = {k: random_locations[k] for k in ('loc-19', )}

        # logger.info("Center")
        # _tmp = {}
        for name, loc in self.center_range.items():
            logger.info(f"{name}: {loc}")
        #     if name in ("loc-0", ):
        #         _tmp[name] = loc
        # self.center_range = _tmp


class SmallConfig(MotifConfig):
    drive = ExternalDrive(5., 30., seeds=np.arange(2))
    AMOUNT_NEURONS = 40,

    PERCENTAGES = .1, -.1
    n_locations = 20
    radius = 5
    # radius = 80

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.5, "base": 301, "octaves": 2, "persistence": .75,},
                            seed=0)


    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(2) #6# Only updating the number, not the values of mean and std.
        generator = np.random.default_rng(seed=0)
        locations = generator.integers(0, self.rows, size=(self.n_locations, 2)).T # 1st location remains the same even for more locations with this style.

        self.center_range = OrderedDict({
            # "repeat-1": (41, 68),
            # "repeat-2": (50, 59),
            # "high-1": (21, 26),
            # "high-2": (48, 71),
            # "gate-1": (15, 33),
            # "gate-2": (0, 10),
            # "start-1": (58, 10),
        })

        random_locations = OrderedDict({f"loc-{i}": tuple(locations[:, i]) for i in range(self.n_locations)})
        self.center_range.update(random_locations)
        # self.center_range = random_locations
        # self.center_range.pop("loc-1", None) # Static bump if -20
        # self.center_range.pop("loc-16", None) # Static bump with +20
        # self.center_range = {k: random_locations[k] for k in ('loc-19', )}

        # logger.info("Center")
        # _tmp = {}
        for name, loc in self.center_range.items():
            logger.info(f"{name}: {loc}")
        #     if name in ("loc-0", ):
        #         _tmp[name] = loc
        # self.center_range = _tmp
