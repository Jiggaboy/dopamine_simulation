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
from cflogger import logger

import numpy as np

from class_lib.population import Population

import lib.dopamine as DOP
import lib.universal as UNI

from params import config

from lib import functimer
from lib.brian import BrianSimulator


@functimer(logger=logger)
def brian():
    force_population = UNI.yes_no("Force to create new population?")
    force_baseline = UNI.yes_no("Force to simulate the baseline?")
    force_patches = UNI.yes_no("Force to simulate the patches?")

    # Sets up a new population. Either loads the connectivity matrix or builds up a new one.
    neural_population = Population(config, force=force_population)
    simulator = BrianSimulator(config, neural_population)
    simulator.run_warmup(force=True)

    for seed in config.drive.seeds:
        simulator.run_baseline(seed, force=force_baseline)
        for radius in config.RADIUSES[:]:
            for name, center in config.center_range.items():
                print(center)
                # Create Patch and retrieve possible affected neurons
                dop_area = DOP.circular_patch(config.rows, center, radius)
                for amount in config.AMOUNT_NEURONS[:]:
                    # Select affected neurons
                    no_of_patches = np.asarray(center).size // 2
                    dop_patch = np.random.choice(dop_area.nonzero()[0], amount * no_of_patches, replace=False)
                    # left_half = dop_patch % config.rows > center[0] # < left, > right
                    # dop_patch = dop_patch[left_half]
                    for percent in config.PERCENTAGES[:]:
                        UNI.log_status(config, radius=radius, name=name, amount=amount, percent=percent)

                        tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
                        simulator.run_patch(dop_patch, percent, tag, seed, force=force_patches)
    return




if __name__ == "__main__":
    brian()
    quit()
