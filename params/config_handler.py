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
from params.motifconfig import MotifConfig, SelectConfig, GateConfig, RepeatConfig, StartConfig, FakeRepeatConfig, RandomLocationConfig, LinkConfig

from collections import OrderedDict
from class_lib import ExternalDrive
import numpy as np
import lib.universal as UNI
from class_lib import Landscape, Synapse, ExternalDrive, TransferFunction
class ExploreConfig(MotifConfig):
    # rows = 120
    transfer_function = TransferFunction(50., .25)
    synapse = Synapse(weight=.3, EI_factor=8.)
    drive = ExternalDrive(2.55, 30., seeds=np.arange(3))
    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1.,
                            connection_probability=.375,
                            params={"size": 2.45, "base": 6, "octaves": 2, "persistence": .5,}, seed=0)

    # Base 16: potential linker

    # base 27: Gate, Start, Repeat
    # base 31: interesting for quadruple intersection
    # base 33: A lot of gating
    # base 34: Static bump as blocker

    # base 50 for repeated simulations

    WARMUP = 200.
    sim_time = 2000.


    PERCENTAGES = .001, .2
    RADIUSES = 80,
    AMOUNT_NEURONS = 1, 50

    center_range = OrderedDict({
        # "link": (40, 40), # base 22: Synchronising
        # "link": (47, 32), # base 24: Synchronising (Radius 8 is too big)
        # "link2": (48, 27), # base 24: Synchronising
        # "link": (64, 35), # base 29: Shutting down a static bump
        # "link": (37, 66), # base 32: Interaction of neighbouring pathways (Global effect visible, but also static bump)
        # "block": (30, 52), # base 34: Blocks a static bump to enable other pathways (global effect?)
        # "link": (20, 39), # base 35: long-range gating? Nope. But some funky global effects.
        # "synchro": (50, 59), # base 35: Synchro neighbouring pathways?
        # "silence": (16, 36),
        "silence": (40, 40),
    })


    def _add_detection_spots(self) -> list:
        detection_spots = []
        # # base 12
        # UNI.append_spot(detection_spots, "start", ((42, 43), (37, 30))) # pre, post
        # base 14
        # UNI.append_spot(detection_spots, "start", ((21, 25), (27, 31))) # pre, post
        # UNI.append_spot(detection_spots, "activate", ((21, 25), (27, 31))) # pre, post
        # base 22
        UNI.append_spot(detection_spots, "link", ((55, 55), (27, 31))) # not done
        UNI.append_spot(detection_spots, "block", ((55, 55), (27, 31))) # not done
        UNI.append_spot(detection_spots, "synchro", ((55, 55), (27, 31))) # not done
        # base
        # UNI.append_spot(detection_spots, "activate", ((37, z), (27, 31))) # pre, post
        # UNI.append_spot(detection_spots, "link", ((21, 25), (27, 31))) # pre, post

        return detection_spots

### Set the current config for all scripts/analyses here:
config = ExploreConfig()
config = SelectConfig()
config = GateConfig()
config = RepeatConfig()
config = FakeRepeatConfig()
config = StartConfig()
config = RandomLocationConfconfig = LinkConfig()
