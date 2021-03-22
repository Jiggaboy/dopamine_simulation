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


import animation.activity as ANIM

import dopamine as DOP

import custom_class.transfer_function as TF




#%% Intialisation


# External input
ext_in_mean = 0
input_std = 4

PATCH_SIZE = 4
THRESHOLD = 0.2
ETA = 1
INTERVAL = 50
DOP_DURATION = 25

# use constant seed for random operations
USE_CONSTANT_SEED = True

PRELOAD_POPULATION = True
SAVE_POPULATION = True                        # only if not preloaded
SAVE_POPULATION_AFTER_SIMULATION = True

CALC_RATE = True
# CALC_RATE = False


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



def simulate(neural_population:Population, **params):
    three_fac_learning = params.get("dopamine_patches")

    taxis = np.arange(CF.sim_time + CF.WARMUP)
    rate = np.zeros((CF.N, taxis.size))


    external_input = np.random.normal(ext_in_mean, input_std, size=rate.shape)

    for t in range(taxis.size-1):
        current_rate = rate[:, t]
        r_dot = neural_population.connectivity_matrix @ current_rate + external_input[:, t]  # expensive!!!
        r_dot = ensure_valid_operation_range(r_dot)

        if t % 450 == 0:
            print(f"{t}: {r_dot.min()} / {r_dot.max()}")

        r_dot = TF.transfer_function(r_dot)
        rate[:, t+1] = current_rate + (- current_rate + r_dot) / CF.TAU
        if three_fac_learning and t > CF.WARMUP:
            if t % INTERVAL == 0:
                # get a dopamine patch
                patch = DOP.perlin_patch(neural_population.grid.width, size=PATCH_SIZE)
                # get active neurons at current timestep t
                exc_neurons = len(neural_population.exc_neurons)
                recently_active_neurons = rate[:exc_neurons, t-DOP_DURATION:t] >= THRESHOLD
                active_neurons = np.any(recently_active_neurons, axis=1)
                W = neural_population.connectivity_matrix
                strengthen_synapses = patch & active_neurons
                strengthen_synapses = np.argwhere(strengthen_synapses).flatten()
                W[strengthen_synapses, :] = W[strengthen_synapses, :] * (1 + ETA)
                # W[:, strengthen_synapses] = W[:, strengthen_synapses] * (1 + ETA) # update out-degrees
    return rate



#%%% Run Simulation and Plot Firing Rates Over Time With Slow Diffusion

neural_population = set_up_population()

if CALC_RATE:
    rate = simulate(neural_population, dopamine_patches=True)
    # rate = simulate(neural_population, dopamine_patches=False)
    PIC.save_rate(rate)
    if SAVE_POPULATION_AFTER_SIMULATION:
        neural_population.save(str(CF.SPACE_WIDTH) + "_final")
else:
    rate = PIC.load_rate()

# neural_population.plot_population()
# neural_population.plot_synapses(800, "y")
# neural_population.plot_synapses(2040, "g")
# neural_population.plot_synapses(N-1, "r")
anim = ANIM.animate_firing_rates(rate, neural_population.coordinates, CF.NE, start=10, step=8)


avgRate = rate[:CF.NE].mean(axis=1)
title = f"Avg. activation: Steepness {CF.STEEPNESS}"
ANIM.activity(avgRate, CF.SPACE_WIDTH, title=title)


import matplotlib.patches as patches


circles = {"in": ((32, 16), "red"),
           "edge": ((32, 10), "orange"),
           "out": ((32, 4), "yellow"),
           }
radius = 4


for key, param in circles.items():
    center, c = param
    circle = patches.Circle(center, radius=radius, color=c, label=key)
    ax = plt.gca()
    p = ax.add_artist(circle)

patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius=radius)


# import custom_class.network_configuration as CN
# CN.plot_shift(*neural_population.coordinates.T, neural_population.shift)

# plt.figure(figsize=(18, 8))
# plt.pcolor(rate[:, ::100])
# colorbar = plt.colorbar()
# plt.xlabel("Time ($10^{-3}$ seconds)")
# plt.ylabel("#neuron")
# plt.title(f"Neurons: {CF.NE}/{CF.NI}; Grid:{CF.SPACE_HEIGHT}x{CF.SPACE_WIDTH}")
# colorbar.set_label('Firing rate')
