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
from datetime import datetime as dt


import os
from brian2 import set_device, device

def thread_baseline(seed, config, population, force:bool):
    pid = os.getpid()
    directory = f"standalone{pid}"
    set_device('cpp_standalone', directory=directory)
    simulator = BrianSimulator(config, population)
    simulator.run_baseline(seed, force=force)
    device.reinit()

def thread_patch(seed, config, population, dop_patch, percent, force:bool, name, radius, amount):
    tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
    pid = os.getpid()
    directory = f"standalone{pid}"
    set_device('cpp_standalone', directory=directory)
    simulator = BrianSimulator(config, population)
    simulator.run_patch(dop_patch, percent, tag, seed, force=force)
    device.reinit()


@functimer(logger=logger)
def brian():
    force_population = UNI.yes_no("Force to create new population?", False)
    force_baseline = UNI.yes_no("Force to simulate the baseline?", False)
    force_patches = UNI.yes_no("Force to simulate the patches?", False)

    # Sets up a new population. Either loads the connectivity matrix or builds up a new one.
    neural_population = Population(config, force=force_population)
    simulator = BrianSimulator(config, neural_population)
    simulator.run_warmup()

    import multiprocessing
    from functools import partial
    with multiprocessing.Pool(6) as p:
        run_sim = partial(thread_baseline, config=config, population=neural_population, force=force_baseline)
        results = p.map(run_sim, config.drive.seeds)


    # for s in config.drive.seeds[:1]:
        # simulator.run_baseline(seed, force=force_baseline)
        for radius in config.RADIUSES[:]:
            for name, center in config.center_range.items():
                # Create Patch and retrieve possible affected neurons
                dop_area = DOP.circular_patch(config.rows, center, radius)
                for amount in config.AMOUNT_NEURONS[:]:
                    # Select affected neurons
                    dop_patch = np.random.choice(dop_area.nonzero()[0], amount, replace=False)
                    # left_half = dop_patch % config.rows > center[0] # < left, > right
                    # dop_patch = dop_patch[left_half]
                    for percent in config.PERCENTAGES[:]:
                        UNI.log_status(config, radius=radius, name=name, amount=amount, percent=percent)

                        # tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
                        # simulator.run_patch(dop_patch, percent, tag, seed, force=force_patches)
                        # executor.submit(thread_patch, config=config, population=neural_population,
                        #                 dop_patch=dop_patch, percent=percent, tag=tag,
                        #                 seed=seed, force=force_patches)
                        # with multiprocessing.Pool(2) as p:
                        run_sim = partial(thread_patch, config=config, population=neural_population,
                                          name=name, radius=radius, amount=amount,
                                          dop_patch=dop_patch, percent=percent, force=force_patches)
                        results = p.map(run_sim, config.drive.seeds)

    # executor.shutdown(wait=True)
    # concurrent.futures.wait(fs, timeout=None, return_when=ALL_COMPLETED)
    return




if __name__ == "__main__":
    brian()
    # for base in np.arange(100, 120):
    #     config.landscape.params["base"] = base
    #     brian()
    quit()
