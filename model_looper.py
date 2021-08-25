# -*- coding: utf-8 -*-
"""
Spreizer Model
"""

import numpy as np

import configuration as CF
import logging
import cflogger
cflogger.set_up()
log = logging.getLogger()



from custom_class.population import Population
import custom_class.pickler as PIC
import custom_class.transfer_function as TF

import dopamine as DOP
import universal as UNI


from time import perf_counter


#%% Intialisation

# use constant seed for random operations
USE_CONSTANT_SEED = True

CALC_RATE = True
# CALC_RATE = False

EXTEND_RATE = True
# EXTEND_RATE = False


MODE = "Perlin_uniform"

center_range = {
    "repeater": (17, 34), # repeater
    "starter": (43, 68), # starter
    "linker": (16, 56), # linker
    "in-activator": (66, 34), # in-activator
    "edge-activator": (63, 34), # edge-activator
    "out-activator": (59, 34), # out-activator
    "in": (35, 18), # in
    "edge": (35, 22), # edge
    "out": (35, 26), # out
}

RADIUSES = (6, 12, 18)
AMOUNT_NEURONS = (10, 50, 100)
PERCENTAGES = (.3, .2, .1)
P_synapses = (1., .8, .6)


before = perf_counter()

def set_up_population():
    try:
        neuron_population = Population.load(CF.SPACE_WIDTH, mode=MODE)
    except FileNotFoundError:
        log.info("Create new Population…", end="")
        neuron_population = Population(CF.NE, CF.NI, grid_size=(CF.SPACE_HEIGHT, CF.SPACE_WIDTH))
        neuron_population.save(CF.SPACE_WIDTH, mode=MODE)
        log.info("Done")

    return neuron_population


def ensure_valid_operation_range(r_dot:np.ndarray, minmax:float=2000.)->np.ndarray:
    r_dot[r_dot > minmax] = minmax
    r_dot[r_dot < -minmax] = -minmax
    return r_dot


def idx2patch(patch):
    """Takes a neuron IDs as input and return a array of exc. neurons """
    null_space = np.zeros(CF.NE, dtype=int)
    null_space[patch] = 1
    return null_space


def init_rate(time, tag:str=None, mode:str=None, force:bool=False)->(np.ndarray, int):
    rate = np.zeros((CF.N, time))
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
        taxis = np.arange(CF.WARMUP)
        rate, start = init_rate(taxis.size, force=True)
    else:
        taxis = np.arange(CF.sim_time + CF.WARMUP)

        try:
            rate, start = init_rate(taxis.size, tag=tag, mode=mode)
        except ValueError:
            return None


    # Generate GWN as ext. input
    external_input = np.random.normal(CF.ext_input_mean, CF.ext_input_std, size=rate.shape)

    for t in range(start, taxis.size-1):
        current_rate = rate[:, t]
        r_dot = neural_population.connectivity_matrix @ current_rate + external_input[:, t]  # expensive!!!
        r_dot = ensure_valid_operation_range(r_dot)

        if t % 400 == 0:
            print(f"{t}: {r_dot.min()} / {r_dot.max()}")

        r_dot = TF.transfer_function(r_dot)
        delta_rate = (- current_rate + r_dot) / CF.TAU

        rate[:, t + 1] = current_rate + delta_rate
    return rate



#%%% Run Simulation and Plot Firing Rates Over Time With Slow Diffusion

neural_population = set_up_population()
EE_matrix_origin = neural_population.connectivity_matrix.copy()

# # Prep
# tag = "_".join((MODE, "warmup"))
# log.info(f"Simulate warmup: {tag}")
# rate = simulate(neural_population, is_warmup=True)
# PIC.save_rate(rate, tag)

# Baseline
tags = UNI.get_filename(MODE, "baseline")
rate = simulate(neural_population, tag=tags, mode=MODE)
PIC.save_rate(rate, tags)



for radius in RADIUSES[:]:
    for name, center in center_range.items():
        dop_area = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
        for amount in AMOUNT_NEURONS[:]:
            dop_patch = np.random.choice(dop_area.nonzero()[0], amount, replace=False)
            for syn in P_synapses:
                dop_patch = np.random.choice(dop_patch, int(dop_patch.size * syn), replace=False)
                for percent in PERCENTAGES[:]:
                    log.info(f"Simulation" \
                          + f" radius: {RADIUSES.index(radius) + 1}/{len(RADIUSES)};"
                          + f" name: {name};"
                          # + f" center: {center_range.index(center) + 1}/{len(center_range)};"
                          + f" amount: {AMOUNT_NEURONS.index(amount) + 1}/{len(AMOUNT_NEURONS)};"
                          + f" syn: {P_synapses.index(syn) + 1}/{len(P_synapses)};"
                          + f" percent: {PERCENTAGES.index(percent) + 1}/{len(PERCENTAGES)};")


                    # And Reset the connectivity matrix here
                    neural_population.connectivity_matrix = EE_matrix_origin.copy()
                    # print(f"Before: {neural_population.connectivity_matrix[:CF.NE, :CF.NE].mean()}")

                    tag_elements = (MODE, name, radius, amount, syn, int(percent*100))
                    tag = UNI.get_filename(tag_elements)
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


# import logging

# LOG_LEVEL = logging.INFO
# LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# LOG_FILE = "parameter.log"
# logging.basicConfig(format=LOG_FORMAT,
#                     handlers=[logging.FileHandler(LOG_FILE),]
#                     )
# logger = logging.getLogger()

# from brian2 import *
# @implementation('cython', '''
#     cdef double torus_distance(double x_pre, double x_post, double y_pre, double y_post):
#         x_pre = x_pre % 1
#         y_pre = y_pre % 1
#         cdef double dx = abs(x_pre - x_post)
#         cdef double dy = abs(y_pre - y_post)

#         if dx > 0.5:
#             dx = 1 - dx

#         if dy > 0.5:
#             dy = 1 - dy

#         return sqrt(dx*dx + dy*dy)
# ''')
# @check_units(x_pre=1, x_post=1, y_pre=1, y_post=1, result=1)
# def torus_distance(x_pre, x_post, y_pre, y_post):
#     x_pre = x_pre % 1
#     y_pre = y_pre % 1

#     dx = abs(x_pre - x_post)
#     dy = abs(y_pre - y_post)

#     if dx > 0.5:
#         dx = 1 - dx

#     if dy > 0.5:
#         dy = 1 - dy

#     return sqrt(dx * dx + dy * dy)
