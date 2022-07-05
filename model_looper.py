# -*- coding: utf-8 -*-
"""
Spreizer Model
"""

import matplotlib.pyplot as plt
import numpy as np

# import configuration as CF
import logging
import cflogger
cflogger.set_up()
log = logging.getLogger()

log.info(f"Logger id: {id(log)}")

from custom_class.population import Population
import util.pickler as PIC

import dopamine as DOP
import universal as UNI

from simulator import Simulator
from params import BaseConfig, TestConfig, PerlinConfig, StarterConfig
Config = TestConfig()
#Config = PerlinConfig()

from time import perf_counter


#%% Intialisation

# use constant seed for random operations
USE_CONSTANT_SEED = True

CALC_RATE = True
# CALC_RATE = False

EXTEND_RATE = True
# EXTEND_RATE = False

before = perf_counter()


def log_status():
    log.info("Simulation" \
          + f" radius: {Config.RADIUSES.index(radius) + 1}/{len(Config.RADIUSES)};"
          + f" name: {name};"
          + f" amount: {Config.AMOUNT_NEURONS.index(amount) + 1}/{len(Config.AMOUNT_NEURONS)};"
          + f" syn: {Config.P_synapses.index(syn) + 1}/{len(Config.P_synapses)};"
          + f" percent: {Config.PERCENTAGES.index(percent) + 1}/{len(Config.PERCENTAGES)};")


#%% Run Simulation and Plot Firing Rates Over Time

# Sets up a new population. Either loads the connectivity matrix or sets up a new one.
neural_population = Population(Config)
# Saves the connectivity matrix for looping simulations
# EE_matrix_origin = neural_population.connectivity_matrix.copy()
# MODE = Config.landscape.mode

UNI.set_seed(USE_CONSTANT_SEED)

## WARMUP
simulator = Simulator(Config, neural_population)
Config.save(subdir=simulator.sub_dir)
simulator.run_warmup()
simulator.run_baseline()


for radius in Config.RADIUSES[:]:
    for name, center in Config.center_range.items():
        # Create Patch and retrieve possible affected neurons
        dop_area = DOP.circular_patch(Config.rows, center, radius)
        for amount in Config.AMOUNT_NEURONS[:]:
            # Select affected neurons
            dop_patch = np.random.choice(dop_area.nonzero()[0], amount, replace=False)
            for syn in Config.P_synapses:
                # TODO: Select % of all the synapses, not of the neurons.
                dop_patch = np.random.choice(dop_patch, int(dop_patch.size * syn), replace=False)
                for percent in Config.PERCENTAGES[:]:
                    log_status()

                    tag = UNI.get_tag_ident(name, radius, amount, syn, int(percent*100))
                    log.info(f"Current tag: {tag}")

                    simulator.run_patch(dop_patch, percent, tag)


after = perf_counter()
log.info(f"Elapsed: {after-before}")

from analysis.analysis import analyze
# analyze()

rate = simulator._load_rate(tag)


from analysis.plot import plot_baseline_activity
plot_baseline_activity(Config)

from animation.activity import animate_firing_rates
anim = animate_firing_rates(rate, neural_population.coordinates, neural_population.exc_neurons.size, start=1, interval=100)
plt.show()
quit()
