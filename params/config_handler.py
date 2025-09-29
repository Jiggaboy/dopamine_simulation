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
from params.motifconfig import RandomLocationConfig

from collections import OrderedDict
import numpy as np
import lib.universal as UNI
from class_lib import Landscape, Synapse, ExternalDrive, TransferFunction

import argparse

parser = argparse.ArgumentParser(description="Select a configuration class.")
parser.add_argument("-c", "--config", type=str, help="Configuration class name")
parser.add_argument("-s", "--seed", type=str, help="Landscape seed")
parser.add_argument("-p", "--percent", type=float, help="Modulation percentage")


args = parser.parse_args()


class ExploreConfig(MotifConfig):
    rows = 30
    warmup = 200.
    sim_time = 1000.
    drive = ExternalDrive(10., 30., seeds=np.arange(2))
    # drive = ExternalDrive(mean, std, seeds=number of seeds=Various GWN instances)
    # ## Simplex noise
    # landscape = Landscape("random", stdE=2.75, stdI=3.)
    # landscape = Landscape("homogeneous", stdE=2.75, stdI=3.)
    # landscape = Landscape("symmetric", stdE=2.75, stdI=3.)

    landscape = Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.45, "base": 25, "octaves": 2, "persistence": .5,}, seed=0)


    center_range = OrderedDict({
        "test": (20, 10),
        "test2": (20, 20),
    })

### Set the current config for all scripts/analyses here:
config = ExploreConfig()
# config = SelectConfig()
# config = GateConfig()
# config = RepeatConfig()
# config = FakeRepeatConfig()
# config = StartConfig()
config = RandomLocationConfig()
# config = CoopConfig()
# config = Gate2Config()

if args.config in globals():
    config = globals()[args.config]()
    print("Configuration loaded :-)")

if args.seed:
    print(f"New seed: {args.seed}")
    config.landscape.seed = int(args.seed)

if args.percent:
    print(f"Updated Percentage: {args.percent}")
    config.PERCENTAGES = UNI.make_iterable(float(args.percent))
