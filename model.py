# -*- coding: utf-8 -*-
"""
Spreizer Model
"""

import numpy as np
import matplotlib.pyplot as plt
import random

import configuration as CF


from custom_class.population import Population
import custom_class.pickler as PIC
import custom_class.transfer_function as TF


import animation.activity as ANIM

import dopamine as DOP





#%% Intialisation

PATCH_SIZE = 4
THRESHOLD = 0.2
INTERVAL = 50
DOP_DURATION = 25

# use constant seed for random operations
USE_CONSTANT_SEED = True

PRELOAD_POPULATION = False
SAVE_POPULATION = True                        # only if not preloaded
SAVE_POPULATION_AFTER_SIMULATION = True

CALC_RATE = True
# CALC_RATE = False

EXTEND_RATE = True
# EXTEND_RATE = False

USE_DOPAMINE_PATCH = True
USE_DOPAMINE_PATCH = False


def set_up_population():
    if USE_CONSTANT_SEED:
        random.seed(0)

    if PRELOAD_POPULATION:
        neuron_population = Population.load(CF.SPACE_WIDTH)
    else:
        neuron_population = Population(CF.NE, CF.NI, grid_size=(CF.SPACE_HEIGHT, CF.SPACE_WIDTH))

    if SAVE_POPULATION and not PRELOAD_POPULATION:
        neuron_population.save(CF.SPACE_WIDTH)

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
        r_dot = ensure_valid_operation_range(r_dot)

        if t % 450 == 0:
            print(f"{t}: {r_dot.min()} / {r_dot.max()}")

        r_dot = TF.transfer_function(r_dot)
        rate[:, t+1] = current_rate + (- current_rate + r_dot) / CF.TAU
        if  t > CF.WARMUP and three_fac_learning:
            if t % INTERVAL == 0:
                # get a dopamine patch
                patch = DOP.perlin_patch(neural_population.grid.width, size=PATCH_SIZE)
                # get active neurons at current timestep t
                exc_neurons = CF.NE
                recently_active_neurons = rate[:exc_neurons, t-DOP_DURATION:t] >= THRESHOLD
                active_neurons = np.any(recently_active_neurons, axis=1)
                strengthen_synapses = patch & active_neurons
                strengthen_synapses = np.argwhere(strengthen_synapses).flatten()
                neural_population.update_synaptic_weights(strengthen_synapses)
                # W[:, strengthen_synapses] = W[:, strengthen_synapses] * (1 + ETA) # update out-degrees
    return rate



#%%% Run Simulation and Plot Firing Rates Over Time With Slow Diffusion

neural_population = set_up_population()

# Perlin scale 5, base=0
circles = {"upper": ((33, 20), "green"),
            "in": ((32, 16), "red"),
            "edge": ((32, 10), "orange"),
            "out": ((32, 4), "yellow"),
            "connector": ((46, 14), "grey"),
            "linker": ((18, 20), "cyan"),
            "repeater": ((7, 24), "cyan"),
            "suppressor3": ((20, 9), "cyan"),

            # latest 40_44
            "enhancer_40_44": ((21, 23), "cyan"),
            # "suppressor_40_44": ((51, 22), "cyan"),

            # latest 40_45
            "enhancer_40_45": ((11, 23), "cyan"),
            "suppressor_40_45": ((51, 22), "cyan"),
            }
radius = 4


# Perlin scale 5, base=1?
# circles = {"upper": ((33, 20), "green"),
#             "in": ((32, 16), "red"),
#             "edge": ((32, 10), "orange"),
#             "out": ((32, 4), "yellow"),
#             "connector": ((24, 41), "grey"),
#             }
# radius = 4


# Perlin scale 2, base=1
circles2 = {"starter": ((20, 28), "grey"),
            "radius": 5
            }

lengthy_patches = [(16, 18),
                   (20, 18),
                   (24, 17),
                   (28, 16),
                   (32, 15),
                   (36, 15),
                   (40, 16),
                   ]

patch = np.full(CF.NE, fill_value=False)
for c in lengthy_patches:
    p = DOP.circular_patch(CF.SPACE_WIDTH, c, radius=2.)
    patch = DOP.merge_patches(patch, p)
# DOP.plot_patch(CF.SPACE_WIDTH, patch)


# patch1 = DOP.circular_patch(CF.SPACE_WIDTH, circles["in"][0], radius=radius)
# patch2 = DOP.circular_patch(CF.SPACE_WIDTH, circles["edge"][0], radius=radius)
# patch3 = DOP.circular_patch(CF.SPACE_WIDTH, circles["upper"][0], radius=radius)
# patch4 = DOP.circular_patch(CF.SPACE_WIDTH, circles["connector"][0], radius=radius)
# patch5 = DOP.circular_patch(CF.SPACE_WIDTH, circles["linker"][0], radius=radius)
patch5 = DOP.circular_patch(CF.SPACE_WIDTH, circles["enhancer_40_44"][0], radius=radius)
patch6 = DOP.circular_patch(CF.SPACE_WIDTH, circles["enhancer_40_44"][0], radius=radius)
# patch7 = DOP.circular_patch(CF.SPACE_WIDTH, circles["connector"][0], radius=radius)
# patch8 = DOP.circular_patch(CF.SPACE_WIDTH, circles["suppressor3"][0], radius=radius)
# patch9 = DOP.circular_patch(CF.SPACE_WIDTH, circles2["starter"][0], radius=circles2["radius"])
# # patch = DOP.merge_patches(patch1, patch2, patch3)

dop_patch = DOP.merge_patches(patch5)
ach_patch = DOP.merge_patches(patch6)
EE_matrix = neural_population.connectivity_matrix[:CF.NE, :CF.NE]
# EE_matrix[dop_patch, :] *= 1.25
EE_matrix[ach_patch, :] *= .8



if CALC_RATE:
    rate = simulate(neural_population, dopamine_patches=USE_DOPAMINE_PATCH)
    PIC.save_rate(rate)
    if SAVE_POPULATION_AFTER_SIMULATION:
        neural_population.save(CF.SPACE_WIDTH, terminated=True)
else:
    rate = PIC.load_rate()


neural_population.plot_indegree()
# neural_population.plot_population()
# neural_population.plot_synapses(3600, "w")
# neural_population.plot_synapses(1620, "y")
# neural_population.plot_synapses(1621, "g")
# neural_population.plot_synapses(1500, "c")
# neural_population.plot_synapses(1501, "r")
# neural_population.plot_synapses(N-1, "r")
anim = ANIM.animate_firing_rates(rate, neural_population.coordinates, CF.NE, start=CF.WARMUP, step=int(CF.TAU / 2.5))


avgRate = rate[:CF.NE].mean(axis=1)
title = f"J: {CF.J}; g: {CF.g}; Perlin scale: {CF.PERLIN_SIZE}"
ANIM.activity(avgRate, title=title, norm=(0, 0.5))

# plt.figure()
# plt.plot(rate[135:150].T)


# neural_population.plot_shift()

import matplotlib.patches as patches

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
