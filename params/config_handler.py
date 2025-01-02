#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Specifies the currently used config!
    Specifies a test-config.

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

from params import BaseConfig
from params.motifconfig import MotifConfig, SelectConfig, GateConfig, RepeatConfig, StartConfig, FakeRepeatConfig
from params.motifconfig import RandomLocationConfig, LinkConfig

from collections import OrderedDict
import numpy as np
import lib.universal as UNI
from class_lib import Landscape, Synapse, ExternalDrive, TransferFunction

class EliasConfig(MotifConfig):
    rows = 36
    synapse = Synapse(weight=.75, EI_factor=7.5)

    transfer_function = TransferFunction(50., .25)
    # synapse = Synapse(weight=.3, EI_factor=7.)
    drive = ExternalDrive(10., 30., seeds=np.arange(2))

    # ## Simplex noise
    landscape = Landscape("simplex_noise", stdE=1.25, stdI=1.5, shift=1.,
                            connection_probability=.375,
                            params={"size": 1., "base": 35,
                                    "octaves": 2, "persistence": .5,},
                            seed=23)

    WARMUP = 200.
    sim_time = 2000.


class ExploreConfig(MotifConfig):
    drive = ExternalDrive(5., 30., seeds=np.arange(2))
    # drive = ExternalDrive(mean, std, seeds=number of seeds=Various GWN instances)
    # ## Simplex noise
    landscape = Landscape("random", stdE=2.75, stdI=3.)
    landscape = Landscape("homogeneous", stdE=2.75, stdI=3.)
    landscape = Landscape("symmetric", stdE=2.75, stdI=3.)
    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 103, "octaves": 2, "persistence": .5,}, seed=0)


### Set the current config for all scripts/analyses here:
config = ExploreConfig()
# config = EliasConfig()
# config = SelectConfig()
# config = GateConfig()
# config = RepeatConfig()
# config = FakeRepeatConfig()
config = StartConfig()
# config = RandomLocationConfig()
# # config = SameNeuronsConfig()
