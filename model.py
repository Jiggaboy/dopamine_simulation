# -*- coding: utf-8 -*-
"""
Spreizer Model
"""

import numpy as np
import matplotlib.pyplot as plt

import configuration as CF


from custom_class.population import Population
import custom_class.pickler as PIC
import custom_class.transfer_function as TF


import animation.activity as ANIM

import dopamine as DOP





#%% Intialisation

# PATCH_SIZE = 4
# THRESHOLD = 0.2
# INTERVAL = 50
# DOP_DURATION = 25

# use constant seed for random operations
USE_CONSTANT_SEED = True

PRELOAD_POPULATION = True
SAVE_POPULATION = True                        # only if not preloaded
SAVE_POPULATION_AFTER_SIMULATION = False

CALC_RATE = True
CALC_RATE = False

EXTEND_RATE = True
# EXTEND_RATE = False

USE_DOPAMINE_PATCH = True
USE_DOPAMINE_PATCH = False

MODE = "independent"
"""Valid options for mode are: Perlin_uniform, symmetric, random, (grid-)independent"""


def set_up_population():
    if USE_CONSTANT_SEED:
        np.random.seed(0)
    else:
        seed = np.random.randint(1000)
        print("Seed:", seed)
        np.random.seed(seed)

    if PRELOAD_POPULATION:
        neuron_population = Population.load(CF.SPACE_WIDTH)
    else:
        neuron_population = Population(CF.NE, CF.NI, grid_size=(CF.SPACE_HEIGHT, CF.SPACE_WIDTH))

    if SAVE_POPULATION and not PRELOAD_POPULATION:
        neuron_population.save(CF.SPACE_WIDTH, mode=MODE)

    return neuron_population


def ensure_valid_operation_range(r_dot:np.ndarray)->np.ndarray:
    VALUE = 2000
    r_dot[r_dot > VALUE] = VALUE
    r_dot[r_dot < -VALUE] = -VALUE
    return r_dot


def init_rate(time)->(np.ndarray, int):
    rate = np.zeros((CF.N, time))
    start = 0

    if EXTEND_RATE:
        try:
            imported_rate = PIC.load_rate()
            start = imported_rate.shape[1] - 1          # account for the connection between preloaded rate and next timestep
            rate[:, :start + 1] = imported_rate
        except FileNotFoundError:
            pass

    return rate, start


def simulate(neural_population:Population, **params):
    three_fac_learning = params.get("dopamine_patches")

    taxis = np.arange(CF.sim_time + CF.WARMUP)
    rate, start = init_rate(taxis.size)

    external_input = np.random.normal(CF.ext_input_mean, CF.ext_input_std, size=rate.shape)

    for t in range(start, taxis.size-1):
        current_rate = rate[:, t]
        r_dot = neural_population.connectivity_matrix @ current_rate + external_input[:, t]  # expensive!!!
        # r_dot = ensure_valid_operation_range(r_dot)

        if t % 350 == 0:
            print(f"{t}: {r_dot.min()} / {r_dot.max()}")

        r_dot = TF.transfer_function(r_dot)
        delta_rate = (- current_rate + r_dot) / CF.TAU
        rate[:, t+1] = current_rate + delta_rate
        # if  t > CF.WARMUP and three_fac_learning:
        #     if t % INTERVAL == 0:
        #         # get a dopamine patch
        #         patch = DOP.perlin_patch(neural_population.grid.width, size=PATCH_SIZE)
        #         # get active neurons at current timestep t
        #         exc_neurons = CF.NE
        #         recently_active_neurons = rate[:exc_neurons, t-DOP_DURATION:t] >= THRESHOLD
        #         active_neurons = np.any(recently_active_neurons, axis=1)
        #         strengthen_synapses = patch & active_neurons
        #         strengthen_synapses = np.argwhere(strengthen_synapses).flatten()
        #         neural_population.update_synaptic_weights(strengthen_synapses)
        #         # W[:, strengthen_synapses] = W[:, strengthen_synapses] * (1 + ETA) # update out-degrees
    return rate



#%%% Run Simulation

neural_population = set_up_population()

circles = {
    # # Perlin scale 5, base=0
    # "upper": ((33, 20), "green"),
    # "in": ((32, 16), "red"),
    # "edge": ((32, 10), "orange"),
    # "out": ((32, 4), "yellow"),
    # "connector": ((46, 14), "grey"),
    # "linker": ((18, 20), "cyan"),
    # # "repeater": ((7, 24), "cyan"),
    # "suppressor3": ((20, 9), "cyan"),

    # # 3-3
    # "linker_33": ((11, 18), "cyan"),
    # "repeater_33": ((18, 40), "cyan"),

    # # 4-0-5-4
    # "repeater_4054": (60, 4),
    # "starter_4054": (17, 16),
    # "linker_4054": (41, 44),
    # "in_4054": (15, 51),
    # "edge_4054": (12, 51),
    # "out_4054": (6, 51),

    # 4-1-5-5
    "repeater_4155": (17, 34),
    "starter_4155": (43, 68),
    "linker_4155": (16, 56),
    "in_4155": (66, 34),
    "edge_4155": (63, 34),
    "out_4155": (59, 34),
    "in_4155_2": (35, 18),
    "edge_4155_2": (35, 22),
    "out_4155_2": (35, 26),
}

radius = 4

patch_rep = DOP.circular_patch(CF.SPACE_WIDTH, circles["repeater_4155"], radius=radius)
patch_start = DOP.circular_patch(CF.SPACE_WIDTH, circles["starter_4155"], radius=radius)
patch_link = DOP.circular_patch(CF.SPACE_WIDTH, circles["linker_4155"], radius=radius)
patch_in = DOP.circular_patch(CF.SPACE_WIDTH, circles["in_4155"], radius=radius)
patch_edge = DOP.circular_patch(CF.SPACE_WIDTH, circles["edge_4155"], radius=radius)
patch_out = DOP.circular_patch(CF.SPACE_WIDTH, circles["out_4155"], radius=radius)
patch_in_2 = DOP.circular_patch(CF.SPACE_WIDTH, circles["in_4155_2"], radius=radius)
patch_edge_2 = DOP.circular_patch(CF.SPACE_WIDTH, circles["edge_4155_2"], radius=radius)
patch_out_2 = DOP.circular_patch(CF.SPACE_WIDTH, circles["out_4155_2"], radius=radius)


dop_patch = DOP.merge_patches(patch_rep)
ach_patch = DOP.merge_patches(patch_link)
no_synapses = dop_patch.nonzero()[0].size
print(no_synapses)
rand_patch = np.random.randint(0, CF.NE, size=no_synapses)
EE_matrix = neural_population.connectivity_matrix[:CF.NE, :CF.NE]

increase = 1.20
decrease = .8
EE_matrix[dop_patch, :] *= increase
EE_matrix[ach_patch, :] *= decrease
# control
# EE_matrix[rand_patch, :] *= increase



if CALC_RATE:
    rate = simulate(neural_population, dopamine_patches=USE_DOPAMINE_PATCH)
    PIC.save_rate(rate)
    if SAVE_POPULATION_AFTER_SIMULATION:
        neural_population.save(CF.SPACE_WIDTH, terminated=True)
else:
    rate = PIC.load_rate()

def idx2patch(patch):
    null_space = np.zeros(CF.NE)
    null_space[patch] = 1
    return null_space

# DOP.plot_patch(CF.SPACE_WIDTH, dop_patch)
DOP.plot_patch(CF.SPACE_WIDTH, idx2patch(rand_patch))
neural_population.plot_indegree()
anim = ANIM.animate_firing_rates(rate, neural_population.coordinates, CF.NE, start=CF.WARMUP, step=int(CF.TAU / 6))


avgRate = rate[:CF.NE].mean(axis=1)
title = f"J: {CF.J}; g: {CF.g}; Perlin scale: {CF.PERLIN_SIZE}"
ANIM.activity(avgRate, title=title, norm=(0, .8), figname="avg_activity")


# neural_population.plot_shift()

#import matplotlib.patches as patches

# for key, param in circles.items():
#     center, c = param
#     circle = patches.Circle(center, radius=radius, color=c, label=key)
#     ax = plt.gca()
#     p = ax.add_artist(circle)

# patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius=radius)


# import custom_class.network_configuration as CN
# CN.plot_shift(*neural_population.coordinates.T, neural_population.shift)

# plt.figure(figsize=(18, 8))
# plt.pcolor(rate[:, ::100])
# colorbar = plt.colorbar()
# plt.xlabel("Time ($10^{-3}$ seconds)")
# plt.ylabel("#neuron")
# plt.title(f"Neurons: {CF.NE}/{CF.NI}; Grid:{CF.SPACE_HEIGHT}x{CF.SPACE_WIDTH}")
# colorbar.set_label('Firing rate')


from time import perf_counter
import dopamine as DOP

before = perf_counter()

RADIUSES = (6, 12, 18)
AMOUNT_NEURONS = (10, 50, 100)
PERCENTAGES = (.3, .2, .1)
for radius in RADIUSES:
    for x in np.linspace(0, CF.SPACE_WIDTH, 10, endpoint=False):
        for y in np.linspace(0, CF.SPACE_WIDTH, 10, endpoint=False):
            patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
            for amount in AMOUNT_NEURONS:
                np.random.choice(patch, amount)
                for percent in PERCENTAGES:
                    EE_matrix = neural_population.connectivity_matrix[:CF.NE, :CF.NE]
                    EE_matrix[dop_patch, :] *= (1. + percent)



after = perf_counter()
print(f"Elapsed: {after-before}")


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
