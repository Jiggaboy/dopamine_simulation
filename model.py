#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Runs the model with different kinds of configurations.

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

import matplotlib.pyplot as plt
import numpy as np

from class_lib.population import Population

import lib.dopamine as DOP
import lib.universal as UNI

from lib.simulator import Simulator
#from NESTsimulator import NESTSimulator
from params import BaseConfig, TestConfig, PerlinConfig
#Config = TestConfig()
Config = PerlinConfig()

from lib import functimer


@functimer(logger=log)
def main():
    # Sets up a new population. Either loads the connectivity matrix or builds up a new one.
    neural_population = Population(Config)

    ## WARMUP
    simulator = Simulator(Config, neural_population)
    Config.save(subdir=simulator.sub_dir)
    simulator.run_warmup()

    for seed in Config.drive.seeds:
        simulator.run_baseline(seed)
        for radius in Config.RADIUSES[:]:
            for name, center in Config.center_range.items():
                # Create Patch and retrieve possible affected neurons
                dop_area = DOP.circular_patch(Config.rows, center, radius)
                for amount in Config.AMOUNT_NEURONS[:]:
                    # Select affected neurons
                    dop_patch = np.random.choice(dop_area.nonzero()[0], amount, replace=False)
                    for percent in Config.PERCENTAGES[:]:
                        log_status(Config, radius=radius, name=name, amount=amount, percent=percent)

                        tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
                        simulator.run_patch(dop_patch, percent, tag, seed)

    return



def log_status(cfg:BaseConfig, radius, name, amount, percent):
    log.info("Simulation" \
          + f" radius: {cfg.RADIUSES.index(radius) + 1}/{len(cfg.RADIUSES)};"
          + f" name: {name};"
          + f" amount: {cfg.AMOUNT_NEURONS.index(amount) + 1}/{len(cfg.AMOUNT_NEURONS)};"
          + f" percent: {cfg.PERCENTAGES.index(percent) + 1}/{len(cfg.PERCENTAGES)};")



if __name__ == "__main__":
    main()
    plt.show()
    quit()
