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
__version__ = '0.2'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

from functools import partial
import numpy as np
import multiprocessing
import shutil
import os

# from lib.neuralhdf5 import NeuralHdf5
from lib.connectivitymatrix import ConnectivityMatrix

from brian2 import set_device, device
import lib.dopamine as DOP
import lib.universal as UNI
from lib.universal import get_neurons_from_patch

from params import config

from lib import functimer
from lib.brian import BrianSimulator

num_processes = 6
brian_dirname = "standalone"


# @functimer(logger=logger)
def brian():
    # If iterate through several shifts:
    # Update the shift in the config as it is used for identification.
    # for base in np.arange(300, 340):
        # config.landscape.params["base"] = base
    for _ in range(1):
        force_population = UNI.yes_no("Force to create new population?", False)
        force_baseline = UNI.yes_no("Force to simulate the baseline?", )
        force_patches = UNI.yes_no("Force to simulate the patches?", )

        # Sets up a new population. Either loads the connectivity matrix or builds up a new one.
        neural_population = ConnectivityMatrix(config, force=force_population)
        # Set up the simulations and connect the neurons.
        
        simulator = BrianSimulator(config, neural_population)
        simulator.run_warmup()
        
        # for seed in config.drive.seeds:
        #     if not force_baseline and simulator.load_rate(config.baseline_tag(seed), no_return=True):
        #         continue
        #     simulator.run_baseline(seed)
        # return
    
        # Parallel simulations.
        with multiprocessing.Pool(processes=num_processes) as p:
            unseen_seeds = config.drive.seeds
            if not force_baseline:
                for seed in config.drive.seeds:
                    # Checks whether simulation was already run before.
                    if simulator.load_rate(config.baseline_tag(seed), no_return=True):
                        unseen_seeds = unseen_seeds[unseen_seeds != seed]
                        continue
    
            run_sim = partial(thread_baseline, config=config)
            _ = p.map(run_sim, unseen_seeds)
            p.close()
            p.join()
            logger.info("Finish baselines")

    # return
    with multiprocessing.Pool(processes=num_processes) as pool:
        simulator = BrianSimulator(config, neural_population)
        for radius in config.radius[:]:
            for name, center in config.center_range.items():
                for amount in config.AMOUNT_NEURONS[:]:
                    for percent in config.PERCENTAGES[:]:
                        print(f"Check {name}...")
                        if not force_patches:
                            unseen_seeds = []
                            for seed in config.drive.seeds:
                                tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
                                if simulator.load_rate(tag, no_return=True):
                                    print(f"Rate {tag} found...")   
                                    continue
                                unseen_seeds.append(seed)
                            unseen_seeds = np.asarray(unseen_seeds)
                        else:
                            unseen_seeds = config.drive.seeds
                        print(f"Unseen seeds: {unseen_seeds}...")  

                        run_sim = partial(thread_patch, config=config,
                                  name=f"{name}", radius=radius, center=center,
                                  amount=amount, percent=percent)

                        _ = [pool.apply_async(run_sim, (seed, )) for seed in unseen_seeds.copy()]
        # Close the pool to prevent more tasks from being submitted
        logger.info("Closing...")
        pool.close()
        logger.info("Simulations finished...")
        # Wait for all worker processes to complete
        # join needs to be calles as apply_async is non-blocking
        pool.join()
        logger.info("Threads terminated...")


@functimer
def thread_baseline(seed:int, config:object) -> None:
    population = ConnectivityMatrix(config)
    simulator = BrianSimulator(config, population)
    simulator.run_baseline(seed)


def thread_patch(seed:int, config:object, name, radius, center, amount, percent):
    # directory = f"{brian_dirname}{os.getpid()}"
    # set_device('cpp_standalone', directory=directory)
    population = ConnectivityMatrix(config)
    simulator = BrianSimulator(config, population)
    dop_area = DOP.circular_patch(config.rows, center, radius)
    # dop_patch = get_neurons_from_patch(dop_area, amount, seed+1) # radius 6
    dop_patch = get_neurons_from_patch(dop_area, amount, int(1e4*center[0] + 1e2*center[1] + seed+1)) # also work for radius 100
    logger.info(f"Neurons: {dop_patch}")
    UNI.log_status(config, radius=radius, name=name, amount=amount, percent=percent)
    tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
    simulator.run_patch(tag, seed, dop_patch, percent)
    # device.reinit()


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
