#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Runs the model based on the brian2 framework with different kinds of configurations.

Description:


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

import cflogger
log = cflogger.getLogger()

import numpy as np

from class_lib.population import Population

import lib.dopamine as DOP
import lib.universal as UNI

from params import BaseConfig, TestConfig, PerlinConfig, StarterConfig, NestConfig
from params import BrianConfig, GateConfig, SelectConfig
Config = SelectConfig()

from lib import functimer
import lib.brian as br

YES = "y"


@functimer(logger=log)
def brian():
    force_population = input("Force to create new population? (y/n)").lower() == YES
    force_baseline = input("Force to simulate the baseline? (y/n)").lower() == YES
    force_patches = input("Force to simulate the patches? (y/n)").lower() == YES

    # Sets up a new population. Either loads the connectivity matrix or builds up a new one.
    neural_population = Population(Config, force=force_population)
    simulator = br.BrianSimulator(Config, neural_population)

    simulator.run_warmup(force=True)

    for seed in Config.drive.seeds:
        simulator.run_baseline(seed, force=force_baseline)
        for radius in Config.RADIUSES[:]:
            for name, center in Config.center_range.items():
                # Create Patch and retrieve possible affected neurons
                dop_area = DOP.circular_patch(Config.rows, center, radius)
                for amount in Config.AMOUNT_NEURONS[:]:
                    # Select affected neurons
                    no_of_patches = np.asarray(center).size // 2
                    dop_patch = np.random.choice(dop_area.nonzero()[0], amount * no_of_patches, replace=False)
                    # left_half = dop_patch % Config.rows > center[0] # < left, > right
                    # dop_patch = dop_patch[left_half]
                    for percent in Config.PERCENTAGES[:]:
                        UNI.log_status(Config, radius=radius, name=name, amount=amount, percent=percent)

                        tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
                        simulator.run_patch(dop_patch, percent, tag, seed, force=force_patches)
    return




if __name__ == "__main__":
    brian()
    quit()
