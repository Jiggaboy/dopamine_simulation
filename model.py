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
brian_dirname = "standalone"

@functimer
def thread_baseline(seed, config, population, force:bool):
    simulator = BrianSimulator(config, population)
    simulator.run_baseline(seed, force=force)


def thread_patch(seed, config, population, force:bool, name, radius, center, amount, percent, dop_patch):
    directory = f"{brian_dirname}{os.getpid()}"
    set_device('cpp_standalone', directory=directory)

    simulator = BrianSimulator(config, population)
    dop_area = DOP.circular_patch(config.rows, center, radius)
    dop_patch = get_neurons_from_patch(dop_area, amount, repeat_samples=False)
    logger.info(f"{dop_patch}")
    UNI.log_status(config, radius=radius, name=name, amount=amount, percent=percent)
    tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
    simulator.run_patch(tag, seed, dop_patch, percent, force=force)
    device.reinit()


@functimer(logger=logger)
def brian():
    # for shift in [.25, .5, .75, 1., 1.5, 2.0, 2.5]:
    #     config.landscape.shift = shift
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
        unseen_seeds = config.drive.seeds
        if not force_patches:
            for seed in config.drive.seeds:
                if simulator.load_rate(config.baseline_tag(seed), no_return=True):
                    unseen_seeds = unseen_seeds[unseen_seeds != seed]
                    continue

        run_sim = partial(thread_baseline, config=config, population=neural_population, force=force_baseline)
        _ = p.map(run_sim, unseen_seeds)
    # return

    with multiprocessing.Pool(processes=num_processes) as pool:
        for radius in config.radius[:]:
            for name, center in config.center_range.items():
                # Create Patch and retrieve possible affected neurons
                dop_area = DOP.circular_patch(config.rows, center, radius)
                for amount in config.AMOUNT_NEURONS[:]:
                    dop_patch = get_neurons_from_patch(dop_area, amount)
                    # Select affected neurons
                    # left_half = dop_patch % config.rows > center[0] # < left, > right
                    # dop_patch = dop_patch[left_half]
                    for percent in config.PERCENTAGES[:]:
                        if not force_patches:
                            unseen_seeds = []
                            for seed in config.drive.seeds:
                                simulator = BrianSimulator(config, neural_population)
                                tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
                                if simulator.load_rate(tag, no_return=True):
                                    continue
                                unseen_seeds.append(seed)
                            unseen_seeds = np.asarray(unseen_seeds)
                        else:
                            unseen_seeds = config.drive.seeds
                        print(unseen_seeds)
                        run_sim = partial(thread_patch, config=config, population=neural_population,
                                  name=f"{name}", radius=radius, center=center, force=force_patches,
                                  amount=amount, percent=percent, dop_patch=dop_patch)
                        # run_sim(seed)
                        # _ = p.map(run_sim, unseen_seeds)
                        _ = [pool.apply_async(run_sim, (seed, )) for seed in unseen_seeds.copy()]
        # Close the pool to prevent more tasks from being submitted
        logger.info("Closing...")
        pool.close()
        # Wait for all worker processes to complete
        pool.join()
        # return


def get_neurons_from_patch(area:np.ndarray, amount:int, repeat_samples:bool=False) -> np.ndarray:
    if repeat_samples:
        np.random.seed(0)
        return np.random.choice(area.nonzero()[0], amount, replace=False)
    else:
        if not hasattr(get_neurons_from_patch, "generator"):
            get_neurons_from_patch.generator = np.random.default_rng()
        return get_neurons_from_patch.generator.choice(area.nonzero()[0], amount, replace=False)
    # To get the same neurons each time



def cleanup_directories():
    for _dir in shutil.os.listdir():
        if _dir.startswith(brian_dirname):
            print("Remove:", _dir)
            shutil.rmtree(_dir)



if __name__ == "__main__":
    brian()
    UNI.play_beep(pause=.5)
    cleanup_directories()
    quit()
