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

from params import BaseConfig, TestConfig, PerlinConfig
Config = TestConfig()
Config = PerlinConfig()

from time import perf_counter


#%% Intialisation

# use constant seed for random operations
USE_CONSTANT_SEED = True

CALC_RATE = True
# CALC_RATE = False

EXTEND_RATE = True
EXTEND_RATE = False


# MODE = "Perlin_uniform"


before = perf_counter()


def init_rate(time, tag:str=None, mode:str=None, force:bool=False)->(np.ndarray, int):
    N = Config.rows**2 + (Config.rows // 2)**2
    rate = np.zeros((N, time))
    start = 0

    if force:
        return rate, start

    if EXTEND_RATE:
        try:
            imported_rate = PIC.load_rate(tag)
            log.info(f"Load rate of: {tag}")
        except FileNotFoundError:
            tag = "_".join((mode, "warmup"))
            imported_rate = PIC.load_rate(tag)
            log.info(f"Load warmup: {tag}")
        start = imported_rate.shape[1] - 1          # account for the connection between preloaded rate and next timestep

        rate[:, :start + 1] = imported_rate

    return rate, start


def simulate(neural_population:Population, **params):
    is_warmup = params.get("is_warmup", False)
    tag = params.get("tag")
    mode = params.get("mode")

    if is_warmup:
        taxis = np.arange(Config.WARMUP)
        rate, start = init_rate(taxis.size, force=True)
    else:
        taxis = np.arange(Config.sim_time + Config.WARMUP)

        try:
            rate, start = init_rate(taxis.size, tag=tag, mode=mode)
        except ValueError:
            return None

    # Immediate return if nothing to simulate
    if start == taxis.size - 1:
        print("Skipped!")
        return rate

    # Generate GWN as ext. input
    external_input = np.random.normal(Config.drive.mean, Config.drive.std, size=rate.shape)

    for t in range(start, taxis.size-1):
        current_rate = rate[:, t]
        r_dot = neural_population.connectivity_matrix @ current_rate + external_input[:, t]  # expensive!!!
        r_dot = UNI.ensure_valid_operation_range(r_dot)

        if t % 400 == 0:
            print(f"{t}: {r_dot.min()} / {r_dot.max()}")

        r_dot = Config.transfer_function.run(r_dot)
        delta_rate = (- current_rate + r_dot) / Config.TAU

        rate[:, t + 1] = current_rate + delta_rate
    return rate



#%% Run Simulation and Plot Firing Rates Over Time

neural_population = Population(Config)



EE_matrix_origin = neural_population.connectivity_matrix.copy()

# plt.show()
# quit()
# # Prep
# tag = "_".join((MODE, "warmup"))
# log.info(f"Simulate warmup: {tag}")
# rate = simulate(neural_population, is_warmup=True)
# PIC.save_rate(rate, tag)

# Baseline
MODE = Config.landscape.mode
tags = UNI.get_tag_ident(MODE, "baseline")
rate = simulate(neural_population, tag=tags, mode=MODE)
PIC.save_rate(rate, tags)

from analysis.analysis import analyze
analyze()


from animation.activity import animate_firing_rates
anim = animate_firing_rates(rate, neural_population.coordinates, neural_population.exc_neurons.size, start=1, interval=100)
plt.show()
quit()


for radius in CF.RADIUSES[:]:
    for name, center in CF.center_range.items():
        dop_area = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
        for amount in CF.AMOUNT_NEURONS[:]:
            dop_patch = np.random.choice(dop_area.nonzero()[0], amount, replace=False)
            for syn in CF.P_synapses:
                dop_patch = np.random.choice(dop_patch, int(dop_patch.size * syn), replace=False)
                for percent in CF.PERCENTAGES[:]:
                    log.info(f"Simulation" \
                          + f" radius: {CF.RADIUSES.index(radius) + 1}/{len(CF.RADIUSES)};"
                          + f" name: {name};"
                          # + f" center: {center_range.index(center) + 1}/{len(center_range)};"
                          + f" amount: {CF.AMOUNT_NEURONS.index(amount) + 1}/{len(CF.AMOUNT_NEURONS)};"
                          + f" syn: {CF.P_synapses.index(syn) + 1}/{len(CF.P_synapses)};"
                          + f" percent: {CF.PERCENTAGES.index(percent) + 1}/{len(CF.PERCENTAGES)};")


                    # And Reset the connectivity matrix here
                    neural_population.connectivity_matrix = EE_matrix_origin.copy()
                    # print(f"Before: {neural_population.connectivity_matrix[:CF.NE, :CF.NE].mean()}")

                    tag_elements = (MODE, name, radius, amount, syn, int(percent*100))
                    tag = UNI.get_tag_ident(*tag_elements)
                    log.info(f"Current tag: {tag}")


                    UNI.set_seed(USE_CONSTANT_SEED)
                    # Update weight matrix
                    EE_matrix = neural_population.connectivity_matrix[:CF.NE, :CF.NE]
                    EE_matrix[dop_patch] *= (1. + percent)
                    # neural_population.connectivity_matrix[:CF.NE, :CF.NE] = EE_matrix

                    # Simulate here
                    rate = simulate(neural_population, tag=tag, mode=MODE)

                    # print(f"{EE_matrix[dop_patch].mean()}")
                    # print(f"After: {neural_population.connectivity_matrix[:CF.NE, :CF.NE].mean()}")
                    PIC.save_rate(rate, tag)

after = perf_counter()
log.info(f"Elapsed: {after-before}")
