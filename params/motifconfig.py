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
    WARMUP = 500
    sim_time = 5000
    rows = 80

    PERCENTAGES = .15,
    RADIUSES = 6,
    AMOUNT_NEURONS = 50,

    transfer_function = TransferFunction(50., .25)
    synapse = Synapse(weight=.3, EI_factor=8.)
    drive = ExternalDrive(10., 30., seeds=np.arange(5))

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.45, "base": 0, "octaves": 2, "persistence": .5,}, seed=0)


class SelectConfig(MotifConfig):
    # drive = ExternalDrive(10., 30., seeds=np.arange(2))
    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.45, "base": 6, "octaves": 2, "persistence": .5,}, seed=0)

    PERCENTAGES = -.1, .1, .2,
    RADIUSES = 6, #8,
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


class Gate(MotifConfig):
    PERCENTAGES = -.2, .1
    RADIUSES = 6,
    AMOUNT_NEURONS = 50,


class LowEffectSizeGateConfig(Gate):
    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.45, "base": 7, "octaves": 2, "persistence": .5,}, seed=0)



    center_range = OrderedDict({
        "gate-left": (17, 43),
        "gate-right": (15, 28),
    })

    def _add_detection_spots(self) -> list:
        detection_spots = []
        center_gate= ((15, 48), (8, 27), (28, 28)) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center_gate)
        UNI.append_spot(detection_spots, "gate-right", center_gate)
        return detection_spots


class GateConfig(Gate):
    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
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


class GateConfig_small(MotifConfig):
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


    def _add_detection_spots(self) -> list:
        detection_spots = []
        center = ((28, 4), (42, 16), (51, 75)) # left, right, merged
        UNI.append_spot(detection_spots, "gate-left", center)
        UNI.append_spot(detection_spots, "gate-right", center)
        return detection_spots


class RepeatConfig(MotifConfig):
    drive = ExternalDrive(5., 30., seeds=np.arange(5))
    PERCENTAGES = -.2, -.1, .1, .2,
    PERCENTAGES = -.2, .2,
    RADIUSES = 6,
    # RADIUSES = 80,
    AMOUNT_NEURONS = 50,
    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.45, "base": 10, "octaves": 2, "persistence": .5,}, seed=0)

    center_range = OrderedDict({
        "repeat": (7, 2),
        "repeat-alt": (7, 2), # difference in detection spots to repeat.
        "repeat-main": (40, 64), # A repeater patch on the main branch
    })


    def _add_detection_spots(self) -> list:
        detection_spots = []
        UNI.append_spot(detection_spots, "repeat", ((11, 31), (10, 11), (9, 69))) # early, pre, post
        UNI.append_spot(detection_spots, "repeat-alt", ((10, 11), (9, 69), (3, 65))) # pre, post, right
        UNI.append_spot(detection_spots, "repeat-main", ((48, 79), (45, 71), (37, 56))) # early, pre, post
        return detection_spots


class FakeRepeatConfig(RepeatConfig):
    PERCENTAGES = .1, .2,
    center_range = OrderedDict({
        "fake-repeat": (37, 59), # later than the main-repeat, establishes a starter in teh second half of the branch.
        # 20% may be to strong, or 1-2 pixels later would work better.
    })


    def _add_detection_spots(self) -> list:
        detection_spots = []
        UNI.append_spot(detection_spots, "fake-repeat", ((48, 79), (45, 71), (37, 56))) # early, pre, post
        return detection_spots


class StartConfig(RepeatConfig):
    drive = ExternalDrive(0., 30., seeds=np.arange(5))
    PERCENTAGES = .1, .2,

    center_range = OrderedDict({
        "start": (45, 12),
    })


    def _add_detection_spots(self) -> list:
        detection_spots = []
        UNI.append_spot(detection_spots, "start", ((44, 8), (46, 76), )) # pre, post
        return detection_spots


# class RandomLocationConfig(MotifConfig):
class RandomLocationConfig(RepeatConfig):
    # drive = ExternalDrive(5., 30., seeds=np.arange(2))
    ## Simplex noise
    # base = 200
    PERCENTAGES = .2, -.2 #-.1, .1, .2,
    n_locations = 6 #20
    RADIUSES = 6
    RADIUSES = 80

    def __post_init__(self):
        super().__post_init__()
        # self.drive.seeds = np.arange(2)
        generator = np.random.default_rng(seed=0)
        locations = generator.integers(0, self.rows, size=(self.n_locations, 2)).T
        self.center_range = OrderedDict({f"loc-{i}": locations[:, i] for i in [2, 3]})
        self.center_range = OrderedDict({f"loc-{i}": locations[:, i] for i in range(self.n_locations)})
        logger.info("Center")
        for c, loc in self.center_range.items():
            logger.info(f"{c}: {loc}")


    def _add_detection_spots(self) -> list:
        detection_spots = []
        loc = ((10, 10), (20, 10))
        for i in range(self.n_locations):
            UNI.append_spot(detection_spots, f"loc-{i}", loc)
        return detection_spots


class SameNeuronsConfig(MotifConfig):
    drive = ExternalDrive(10., 30., seeds=np.arange(5))
    # ## Simplex noise
    base = 300
    n_locations = 10 #20
    RADIUSES = 80
    AMOUNT_NEURONS = 50, 100,

    def __post_init__(self):
        super().__post_init__()
        generator = np.random.default_rng(seed=0)
        locations = generator.integers(0, self.rows, size=(self.n_locations, 2)).T
        self.center_range = OrderedDict({f"loc-{i}": locations[:, i] for i in range(self.n_locations)})


    def _add_detection_spots(self) -> list:
        detection_spots = []
        loc = ((10, 10), (20, 10))
        for i in range(self.n_locations):
            UNI.append_spot(detection_spots, f"loc-{i}", loc)
        return detection_spots


class LinkConfig(MotifConfig):
    base = 22
    drive = ExternalDrive(5., 30., seeds=np.arange(5))

    PERCENTAGES = .1,
    RADIUSES = 8,
    AMOUNT_NEURONS = 30, 50

    center_range = OrderedDict({
        "link": (40, 40),
    })
