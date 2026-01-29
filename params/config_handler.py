# !/usr/bin/env python3
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
from params.motifconfig import RandomLocationConfig, LocationConfig, RandomConfig

from collections import OrderedDict
import numpy as np
import lib.universal as UNI
from class_lib import Landscape, Synapse, ExternalDrive, TransferFunction

import argparse

parser = argparse.ArgumentParser(description="Select a configuration class.")
parser.add_argument("-c", "--config", type=str, help="Configuration class name")
parser.add_argument("-s", "--seed", type=str, help="Landscape seed")
parser.add_argument("-b", "--base", type=str, help="Landscape base")
parser.add_argument("-p", "--percent", type=float, help="Modulation percentage")


args = parser.parse_args()


class ExploreConfig(MotifConfig):
    rows = 80
    warmup = 250.
    sim_time = 1000.
    drive = ExternalDrive(5., 30., seeds=np.arange(1))
    synapse = Synapse(weight=.3, EI_factor=8.)
    # drive = ExternalDrive(mean, std, seeds=number of seeds=Various GWN instances)

    ### Simplex noise
    # landscape = Landscape("random", stdE=2.75, stdI=3.)
    # landscape = Landscape("homogeneous", stdE=2.75, stdI=3.)
    # landscape = Landscape("symmetric", stdE=2.75, stdI=3.)

    # size 1, base 20 has good gates
    # size 1, base 21 has a single pathway
    landscape = Landscape("simplex_noise", stdE=1.25, stdI=2, shift=1., connection_probability=.75,
                            params={"size": 1, "base": 21, "octaves": 1, "persistence": .5,}, seed=0)


    # Base 23, size 2.3 potential select in the bottom right
    landscape = Landscape("simplex_noise", stdE=1., stdI=2, shift=1., connection_probability=.75,
                            params={"size": 2.3, "base": 21, "octaves": 1, "persistence": .5,}, seed=0)
    landscape = Landscape("simplex_noise", stdE=1.5, stdI=2, shift=1., connection_probability=.75,
                            params={"size": 2.3, "base": 30,}, seed=0)

    
    # Top right
    # size 2.3 favors right
    # Default octaves and persistence (2/0.5)
    # persistence 0.6 is better, but has a static bump (which is there at 0.5 already)
    # persistence 0.4 kills left
    # persistence 0.75 separates the right branch
    # Octaves 3 does not kill the static bump
    # stdI 2.8 kills left
    # stdE 2.5 and stdI 2.8 revives it
    drive = ExternalDrive(5., 30., seeds=np.arange(1))
    synapse = Synapse(weight=.3, EI_factor=8.)
    landscape = Landscape("simplex_noise", stdE=2.6, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.25, "base": 116, "octaves": 2, "persistence": .5,}, seed=0)
    
    
    # # GREAT SELECT MOTIF ??
    landscape = Landscape("simplex_noise", stdE=2.6, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.25, "base": 114,}, seed=0)


    # Well....
    # landscape = Landscape("simplex_noise", stdE=2.5, stdI=3., shift=1., connection_probability=.375,
    #                         params={"size": 2.45, "base": 48, "octaves": 2, "persistence": 0.6,}, seed=0)


    # # Good select motif: But persistent acitivty that the branch cannot be selected...
    drive = ExternalDrive(5., 30., seeds=np.arange(1))
    synapse = Synapse(weight=.3, EI_factor=8.)
    landscape = Landscape("simplex_noise", stdE=2.5, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.4, "base": 23,
                                    "octaves": 2, "persistence": 0.5,}, seed=1)
    
    rows = 100
    drive = ExternalDrive(0., 40., seeds=np.arange(1))
    synapse = Synapse(weight=.3, EI_factor=8.)
    landscape = Landscape("simplex_noise", stdE=2.5, stdI=2.5, shift=1., connection_probability=.375,
                            params={"size": 2.4, "base": 23,
                                    "octaves": 2, "persistence": 0.5,}, seed=1)
    drive = ExternalDrive(0., 30., seeds=np.arange(1))
    synapse = Synapse(weight=.275, EI_factor=8.)
    # Going from stdE 2.35 to 2.5 reduces activity a lot - but keeps the select
    landscape = Landscape("simplex_noise", stdE=2.5, stdI=2.5, shift=1., connection_probability=.375,
                            params={"size": 2.4, "base": 23,
                                    "octaves": 2, "persistence": 0.5,}, seed=1)
    
    
    # SelectConfig
    # drive = ExternalDrive(10., 30., seeds=np.arange(1))
    # synapse = Synapse(weight=.3, EI_factor=8.)
    # landscape = Landscape("simplex_noise", stdE=2.65, stdI=3., shift=1., connection_probability=.375,
    #                         params={"size": 2.5, "base": 6, "octaves": 2, "persistence": .5,}, seed=2)



### Set the current config for all scripts/analyses here:
config = ExploreConfig()
# config = SelectConfig()
# config = GateConfig()
# config = RepeatConfig()
# config = FakeRepeatConfig()
# config = StartConfig()
config = RandomLocationConfig()
config = LocationConfig()
# config = CoopConfig()
config = ExploreConfig()
config = SelectConfig()
# config = GateConfig()
# config = RandomConfig()

if args.config in globals():
    config = globals()[args.config]()
    print("Configuration loaded :-)")

if args.seed:
    print(f"New seed: {args.seed}")
    config.landscape.seed = int(args.seed)

if args.base:
    print(f"New seed: {args.base}")
    config.landscape.params["base"] = int(args.base)

if args.percent:
    print(f"Updated Percentage: {args.percent}")
    config.PERCENTAGES = UNI.make_iterable(float(args.percent))
