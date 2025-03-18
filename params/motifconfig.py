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
    drive = ExternalDrive(10., 30., seeds=np.arange(4))

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
    PERCENTAGES = .1, .2
    radius = 6,
    AMOUNT_NEURONS = 50,
    save_synaptic_input = True

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 9, "octaves": 2, "persistence": .5,}, seed=0)

    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(2)

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



class CoopConfig(GateConfig):
    PERCENTAGES = .1, .25, -.2

    save_synaptic_input = True

    landscape = Landscape("simplex_noise", stdE=2.8, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 9, "octaves": 2, "persistence": .5,}, seed=1)

    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(2)


class Gate2Config(MotifConfig):
    # sim_time = 2000.
    drive = ExternalDrive(5., 30., seeds=np.arange(2))
    PERCENTAGES = .2, -.25
    save_synaptic_input = True

    center_range = OrderedDict({
        "gate-left": (71, 11),
        "gate-right": (56, 10),
        # "gate-right": (67, 25),
    })

    landscape = Landscape("simplex_noise", stdE=2.8, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 23, "octaves": 2, "persistence": .5,}, seed=0)

    def _add_detection_spots(self) -> list:
        detection_spots = []
        # center_gate= ((2, 21), (60, 30), (73, 11)) # left, right, merged
        # UNI.append_spot(detection_spots, "gate-right", center_gate)
        center_gate= ((73, 11), (55, 13), (74, 1)) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center_gate)
        UNI.append_spot(detection_spots, "gate-right", center_gate)
        return detection_spots


class Gate3Config(MotifConfig):
    sim_time = 2000.

    PERCENTAGES = .15, -.25, .25
    save_synaptic_input = True

    center_range = OrderedDict({
        # "gate-left": (43, 26),
        # "gate-right": (40, 16),
        "gate-left": (43, 27),
        "gate-right": (37, 17),
    })

    landscape = Landscape("simplex_noise", stdE=2.8, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 25, "octaves": 2, "persistence": .5,}, seed=0)

    def _add_detection_spots(self) -> list:
        detection_spots = []
        center_gate= ((42, 32), (30, 16), (50, 8)) # left, right, merged
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
    radius = 80

    def __post_init__(self):
        super().__post_init__()
        self.drive.seeds = np.arange(6) # Only updating the number, not the values of mean and std.
        generator = np.random.default_rng(seed=0)
        locations = generator.integers(0, self.rows, size=(self.n_locations, 2)).T # 1st location remains the same even for more locations with this style.

        self.center_range = OrderedDict({
            "start": (45, 12),
            "repeat": (7, 2),
            "repeat-main": (40, 64),
            "fake-repeat": (37, 57), # later than the main-repeat, establishes a starter in the second half of the branch.
            # "anti-repeat": (39, 54),
        })

        random_locations = OrderedDict({f"loc-{i}": locations[:, i] for i in range(self.n_locations)})
        self.center_range.update(random_locations)
        self.center_range = random_locations

        logger.info("Center")
        for name, loc in self.center_range.items():
            logger.info(f"{name}: {loc}")
