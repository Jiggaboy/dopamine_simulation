#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:37:12 2021

@author: hauke
"""

import numpy as np
import matplotlib.pyplot as plt

from peakutils import peak as putils

import configuration as CF
import util.pickler as PIC
import dopamine as DOP
import universal as UNI


### Idea: Plot the number of sequences across multiple simulations

keys, space = UNI.get_parameter_space()
c_idx = list(keys).index("center")
r_idx = list(keys).index("radius")
n_idx = list(keys).index("n_neurons")
m_idx = list(keys).index("m_synapses")
p_idx = list(keys).index("p_weight")

space_name = "linker"
linker_space = UNI.sort_space(space, space_name)

# TODO: Add a baseline level as first data point
# TODO: Add some discrimation line?
# TODO: Reduce the number of data points to a specific parameter of interest.
# TODO: Uniform scale


MODE = "Perlin_uniform"
bs_tag = UNI.get_tag_ident(MODE, "baseline")

center_pre = (21, 65)
center_post = (30, 63)
center_3 = (23, 44)
center = (center_pre, center_post, center_3, )
radius = 2



def main():
    s_bs = passing_sequences(center, radius, bs_tag)

    for i, p in enumerate(linker_space):
        p = p.tolist()
        p[r_idx - 1] = int(p[r_idx - 1])
        p[n_idx - 1] = int(p[n_idx - 1])
        p[p_idx - 1] = int(p[p_idx - 1])
        tags = UNI.get_tag_ident(MODE, space_name, *p)
        s = passing_sequences(center, radius, tags)

        for j, c in enumerate(center):
            plt.figure(str(c))
            plt.plot(i, s[1][j], marker="*")
        # if i >= 3:
        #     break

    for j, c in enumerate(center):
        plt.figure(str(c))
        plt.axhline(s_bs[1][j], label="baseline")
        plt.ylim(0, .015)
        plt.gca().set_xticks(np.arange(i))
        plt.gca().set_xticklabels(linker_space[:i], rotation='vertical', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.title(f"linker: patch @ {c}")


def plot_passing_sequences_pre_post(patches:np.ndarray, postfix:str, figname:str, title:str=None, details:tuple=None, details_in_title:bool=True):
    from analysis.passing_sequences import number_of_sequences
    # Resetting the color cycle, why?
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    width = 2.
    weigth = 5
    plt.figure(figname, figsize=(4, 2.8))

    counts = [number_of_sequences(p.nonzero()[0], avg=False, postfix=postfix) for p in patches]

    heights, bins, handlers_neurons = plt.hist(counts, bins=np.arange(0, 250, 10), color=[colors[0], colors[2]])
    handlers_avg = []
    for idx, p in enumerate(patches):
        c_idx = 2 * idx + 1
        count = number_of_sequences(p, avg=True, postfix=postfix)
        print(f"{postfix}: {count} - {idx}")
        _, _, handler = plt.hist(count, color=colors[c_idx], bins=[count-width, count+width], weights=[weigth])
        handlers_avg.append(handler)
    plt.title(title)
    handlers = (
        handlers_neurons[0],
        handlers_avg[0],
        handlers_neurons[1],
        handlers_avg[1],
        )
    # handlers = handlers_neurons
    labels = (
        "Main: Ind. neurons",
        "Main: Avg. activity",
        "Low. pathway: Ind. neurons",
        "Low. pathway: Avg. activity",
        )
    # repeater
    # plt.xlim(0, 255)
    # plt.ylim(0, 11)
    # linker
    # plt.xlim(0, 210)
    # plt.ylim(0, 12)
    # activator
    plt.xlim(0, 250)
    plt.ylim(0, 11)
    plt.legend(handlers, labels)


def passing_sequences(center, radius, tag:str):
    patches = [DOP.circular_patch(CF.SPACE_WIDTH, c, radius) for c in center]
    neurons = [UNI.patch2idx(p) for p in patches]

    rate = PIC.load_rate(tag, exc_only=True, skip_warmup=True)
    counts = [number_of_sequences(n, avg=False, rate=rate) for n in neurons]
    avg_counts = [number_of_sequences(n, avg=True, rate=rate) for n in neurons]
    return counts, avg_counts


def number_of_sequences(neuron:(int, iter), avg:bool=False, rate:np.ndarray=None, threshold:float=None, min_dist:int=None, normalize:bool=True)->int:
    """
    Detect the number of sequences for the given neuron(s) for a given rate. Depending on the paramter threshold and min_dist.

    Parameters
    ----------
    neuron : (int, iterable)
        The neuron(s) to be analyzed.
    avg : bool, optional
        Average across neurons e.g. for a patch of neurons. The default is False.
    postfix : str, optional
        Use different rate-file. The default is None.
    threshold : float, optional
        Passed to the peak detection. The default is defined in the method.
    min_dist : int, optional
        Passed to the peak detection. The default is CF.TAU.

    Returns
    -------
    int
        DESCRIPTION.

    """
    threshold = threshold or 0.2
    min_dist = min_dist or CF.TAU

    if isinstance(neuron, int):
        number =  number_of_peaks(rate[neuron], thres=threshold, min_dist=min_dist)
    else:
        if avg:
            number =  number_of_peaks(rate[neuron].mean(axis=0), thres=threshold, min_dist=min_dist)
        else:
            number = np.zeros(len(neuron))
            for idx, n in enumerate(neuron):
                number[idx] = number_of_peaks(rate[n], thres=threshold, min_dist=min_dist)

    if normalize:
        number = number / rate.shape[1]
    return number



def number_of_peaks(data, **kwargs):
    no = putils.indexes(data, **kwargs, thres_abs=True).size
    return no


if __name__ == "__main__":
    main()
