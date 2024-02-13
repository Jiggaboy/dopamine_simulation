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

from functools import partial
import numpy as np
import multiprocessing
import shutil
import os

from class_lib.population import Population

from brian2 import set_device, device
import lib.dopamine as DOP
import lib.universal as UNI

from params import config

from lib import functimer
from lib.brian import BrianSimulator

num_processes = 6

@functimer
def thread_baseline(seed, config, population, force:bool):
    pid = os.getpid()
    directory = f"standalone{pid}"
    set_device('cpp_standalone', directory=directory)
    simulator = BrianSimulator(config, population)
    simulator.run_baseline(seed, force=force)
    device.reinit()

def thread_patch(seed, config, population, force:bool, name, radius, center):
    pid = os.getpid()
    directory = f"standalone{pid}"
    set_device('cpp_standalone', directory=directory)
    simulator = BrianSimulator(config, population)
    dop_area = DOP.circular_patch(config.rows, center, radius)
    for amount in config.AMOUNT_NEURONS:
        dop_patch = get_neurons_from_patch(dop_area, amount)
        for percent in config.PERCENTAGES:
            UNI.log_status(config, radius=radius, name=name, amount=amount, percent=percent)
            tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
            simulator.run_patch(tag, seed, dop_patch, percent, force=force)
    device.reinit()


def cleanup_directories():
    for _dir in shutil.os.listdir():
        if _dir.startswith("standalone"):
            print("Remove:", _dir)
            shutil.rmtree(_dir)



@functimer(logger=logger)
def brian():
    force_population = UNI.yes_no("Force to create new population?", False)
    force_baseline = UNI.yes_no("Force to simulate the baseline?", False)
    force_patches = UNI.yes_no("Force to simulate the patches?", False)

    # Sets up a new population. Either loads the connectivity matrix or builds up a new one.
    neural_population = Population(config, force=force_population)
    simulator = BrianSimulator(config, neural_population)
    simulator.run_warmup()

    # for seed in config.drive.seeds:
    #     thread_baseline(config=config, population=neural_population, force=force_baseline, seed=seed)
    with multiprocessing.Pool(processes=num_processes) as p:
        run_sim = partial(thread_baseline, config=config, population=neural_population, force=force_baseline)
        results = p.map(run_sim, config.drive.seeds)

        for radius in config.RADIUSES[:]:
            for name, center in config.center_range.items():
                # # Create Patch and retrieve possible affected neurons
                # dop_area = DOP.circular_patch(config.rows, center, radius)
                # for amount in config.AMOUNT_NEURONS[:]:
                #     dop_patch = get_neurons_from_patch(dop_area, amount)
                    # Select affected neurons
                    # left_half = dop_patch % config.rows > center[0] # < left, > right
                    # dop_patch = dop_patch[left_half]
                    # for percent in config.PERCENTAGES[:]:
                run_sim = partial(thread_patch, config=config, population=neural_population,
                                  name=f"{name}", radius=radius, center=center, force=force_patches)
                results = p.map(run_sim, config.drive.seeds)
    return


def get_neurons_from_patch(area:np.ndarray, amount:int) -> np.ndarray:
    return np.random.choice(area.nonzero()[0], amount, replace=False)



if __name__ == "__main__":
    brian()
    # for base in np.arange(100, 120):
    #     config.landscape.params["base"] = base
    #     brian()
    UNI.play_beep(pause=.5)
    cleanup_directories()
    quit()
