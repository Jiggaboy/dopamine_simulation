#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:38:43 2021

@author: hauke
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as sk

from custom_class.population import Population
import configuration as CF

import custom_class.pickler as PIC

import animation.activity as ACT



steep = ["10",
         "1_0",
         "0_1",]

rates = {}
for s in steep:
    rate = PIC.load_rate(s)
    avgRate = rate[:CF.NE].mean(axis=1)
    rates[s] = avgRate
    ACT.activity(avgRate, CF.SPACE_WIDTH, title=f"{s}")


high = []
for steep_i, rate_i in rates.items():
    high.append(steep_i)
    for steep_j, rate_j in rates.items():
        if steep_j in high:
            continue
        rate_diff = rate_i - rate_j
        ACT.activity(rate_diff, CF.SPACE_WIDTH, title=f"{steep_i} - {steep_j}", norm=(None, None))



def analyze_anatomy():
    # fig, axes = plt.subplots(1, 2)
    populations = []
    populations.append((Population.load(CF.SPACE_WIDTH), "b", "c"))
    populations.append((Population.load(str(CF.SPACE_WIDTH) + "_final"), "y", "orange"))

    populations[0][0].plot_population()
    neuron_base = np.random.randint(populations[0][0].exc_neurons.size)
    # neuron_base = 438
    # neuron_base = 2489
    print(f"Neuron: {neuron_base}")

    for population_set in populations:
        population = population_set[0]

        W = population.connectivity_matrix


        # Select neurons
        gridsize = 2
        total_neurons = gridsize**2
        neuron_slices = []
        for i in range(gridsize):
            right = neuron_base - (i * CF.SPACE_WIDTH) + 1
            left = right - gridsize
            if left % CF.SPACE_WIDTH > right % CF.SPACE_WIDTH:
                row = population.coordinates[right][1]
                neuron_slices.append(slice(left + CF.SPACE_WIDTH, (row + 1) * CF.SPACE_WIDTH))
                neuron_slices.append(slice(row * CF.SPACE_WIDTH, right))
            else:
                neuron_slices.append(slice(left, right))

        idcs = []
        for sl in neuron_slices:
            plt.scatter(*population.coordinates[sl].T, c="g")
            idx = range(*sl.indices(population.exc_neurons.size))
            idcs.extend(list(idx))
        start = idcs

        all_connected_neurons = set()
        for _ in range(100):
            condensed_W = W[idcs, :]
            condensed_out_degree = condensed_W.sum(axis=0)
            idcs = condensed_out_degree.argsort()[-total_neurons:][::-1]

            all_connected_neurons.update(idcs)

        # print(f"Starting neurons: {start}")
        # print(f"Finishing neurons: {idcs}")

        distances = []
        for i in range(total_neurons):
            for j in range(total_neurons):
                distance = population.grid.get_distance(population.coordinates[start[i]], population.coordinates[idcs[j]])
            distances.append(distance)
        print(np.mean(distances))

        plt.scatter(*population.coordinates[sorted(all_connected_neurons)].T, c=population_set[1])
        plt.scatter(*population.coordinates[start[0]].T, c="g")
        plt.scatter(*population.coordinates[idcs].T, c=population_set[2])
        plt.title("Neurons involved in aSTAS")


def hist_activity():
    da = 0.1

    dop_rate = PIC.load_rate(postfix="dop")
    dop_rate = dop_rate[CF.WARMUP:]

    baseline_rate = PIC.load_rate(postfix="baseline")
    baseline_rate = baseline_rate[CF.WARMUP:]

    out_rate = PIC.load_rate(postfix="out-degree")
    out_rate = out_rate[CF.WARMUP:]

    all_rates = np.asarray([baseline_rate.flatten(),
                            dop_rate.flatten(),
                            # out_rate.flatten(),
                            ])

    plt.hist(all_rates.T, bins=np.arange(da, 1+da, da))
    plt.legend(["baseline", "dopamine - in", "dopamine - out"])


def run_PCA():
    setups = ["pca_baseline",
              "pca_dopamine",
              "pca_dop_2",
              ]

    for s in setups:
        rate = PIC.load_rate(postfix=s)

        # n_samples x n_features
        try:
            pca = PIC.load(s)
        except FileNotFoundError:
            pca = sk.PCA()
            pca.fit(rate.T)
            PIC.save(s, pca)
        cumsumVariances = sum_variances(pca.explained_variance_ratio_)
        plot_explained_variance_ratio(cumsumVariances, s)
        plt.legend()


def plot_explained_variance_ratio(data:tuple, lbl:str):
    plt.figure("Explained Variance")
    plt.title("Explained Variance as function of PCs")

    if not plt.gca().lines:
        plt.axhline(0.9, color="red", ls="--", label="90%")
        plt.axhline(0.7, color="green", ls="--", label="70%")

    plt.plot(*data, label=lbl)
    plt.xlabel("PCs")
    plt.ylabel("Explained variance")
    plt.ylim([0., 1.05])
    plt.tight_layout()


def sum_variances(explained_variance_ratio:np.ndarray)->tuple:
    cumRatio = []
    for idx, el in enumerate(explained_variance_ratio):
        try:
            cumRatio.append(cumRatio[idx-1] + el)
        except IndexError:
            cumRatio.append(el)
    cumRatio = np.asarray(cumRatio)
    xRange = range(1, len(cumRatio)+1)

    return (xRange, cumRatio)


if __name__ == "__main__":
    # hist_activity()
    # run_PCA()
    # analyze_anatomy()
    pass
