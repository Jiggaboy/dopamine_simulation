#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:38:43 2021

@author: hauke
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as sk

from peakutils import peak as putils

from custom_class.population import Population
import configuration as CF
import dopamine as DOP

import custom_class.pickler as PIC

import animation.activity as ACT
import animation.rate as RAT

def analyze():
    # 1 Making a histogram of the rates of the neurons.
    # rate = ["baseline", "dop", "out-degree"]
    # rate_labels = ["baseline", "in-degree", "out-degree"]
    # hist_activity(rate, rate_labels)

    # 2 Running a PCA on the rates.
    rate_postfixes = ["bs", "in", "edge", "out",]
    # rate_postfixes = ["pca_baseline", "pca_dopamine", "pca_dop_2",]
    # run_PCA(rate_postfixes, force=True)

    center, radius = (32, 16), 6
    patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
    block_PCA("bs", "in", force=True)
    # block_PCA("bs", "in", patch, force=True)

    pass


def analyze_circular_dopamine_patch():
    setups = ["in+edge",
              "in",
              "edge",
              "out",
              "bs",
              ]
    setups = ["g5_5",
              "g6_0",
              "g7_0",
              "g8_0",
              ]
    # setups = ["J4_0",
    #           "J3_0",
    #           "J2_0",
    #           # "g8_0",
    #           ]
    setups = [
              # "linker",
              # "linker_ACh",
              "repeater",
              "low_activity",
              # "g6_0_J2_5",
              ]
    setups = [
              # "linker",
              "P2_25p",
              "P2_35p",
              "P2_bs",
              ]



    rates = merge_avg_rate_to_key(setups, plot=True)
    plot_rate_differences(rates, norm=(-.3, .3))


def merge_avg_rate_to_key(keys:list, plot:bool=False)->dict:
    rates = {}
    for s in keys:
        rate = PIC.load_rate(s)
        avgRate = rate[:CF.NE].mean(axis=1)
        rates[s] = avgRate
        if plot:
            ACT.activity(avgRate, CF.SPACE_WIDTH, title=f"{s}", figname=f"circ_patch_{s}")
    return rates


def plot_rate_differences(avg_rates:dict, norm:tuple=None):
    norm = norm or (None, None)

    high = []
    for key_i, rate_i in avg_rates.items():
        high.append(key_i)
        for key_j, rate_j in avg_rates.items():
            if key_j in high:
                continue
            rate_diff = rate_i - rate_j
            # To be adjusted
            figname = f"circ_patch_{key_i}_{key_j}"
            title = f"{key_i} - {key_j}: {rate_diff.mean():.5f}"
            ACT.activity(rate_diff, CF.SPACE_WIDTH, figname=figname, title=title, norm=norm, cmap=plt.cm.seismic)


def analyze_travel_direction(patch:np.ndarray, patchdetails:tuple, postfix:str=None, delta_t:float=None, threshold:float=None, plot_rates:bool=True):
    threshold = threshold or 0.3
    delta_t = delta_t or 10

    # load rate
    rate = PIC.load_rate(postfix, skip_warmup=True, exc_only=True)

    mean_rate = rate[patch].mean(axis=0)
    # Check
    if plot_rates:
        RAT.rate(rate[patch], avg=True, threshold=threshold)

    # Get crossings above threshold and avoid IndexErrors by cutting results that would be after end of simulation.
    crossings = np.where(mean_rate > threshold)[0]
    crossings = crossings[crossings + delta_t < rate.shape[1]]
    snapshot_pre = rate[:, crossings].mean(axis=1)
    snapshot_post = rate[:, crossings + delta_t].mean(axis=1)

    title = f"Snapshot @{patchdetails[0]} with r={patchdetails[1]}"
    title_post = title + f"\n Delta t: {delta_t}ms"
    des =  {"title_pre": title,
            "title_post": title_post,}

    ACT.pre_post_activity(snapshot_pre, snapshot_post, **des)


def analyze_steepness():
    steep = ["10",
             "1_0",
             "0_1",]

    rates = merge_avg_rate_to_key(steep, plot=True)

    high = []
    for steep_i, rate_i in rates.items():
        high.append(steep_i)
        for steep_j, rate_j in rates.items():
            if steep_j in high:
                continue
            rate_diff = rate_i - rate_j
            ACT.activity(rate_diff, CF.SPACE_WIDTH, title=f"{steep_i} - {steep_j}", norm=(None, None))



def analyze_anatomy():
    populations = []
    populations.append((Population.load(CF.SPACE_WIDTH), "b", "c"))
    populations.append((Population.load(CF.SPACE_WIDTH, terminated=True), "y", "orange"))

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


def block_PCA(baseline:str, conditional:str, patch:np.ndarray=None, force:bool=False):

    # n = rate.shape[0] // 2
    # peaks = putils.indexes(rate[n], thres=.3, min_dist=CF.TAU, thres_abs=True)
    # print(peaks)
    # print(peaks[peaks > 10800].size)
    # print(peaks[peaks < 10800].size)

    components = 3

    bs_rate = PIC.load_rate(postfix=baseline, skip_warmup=True, exc_only=True)
    c_rate = PIC.load_rate(postfix=conditional, skip_warmup=True, exc_only=True)

    if patch is None:
        is_patch = False
        patch = np.full(bs_rate.shape[0], fill_value=True)
        subsets = {"global": patch,}
    else:
        is_patch = True
        subsets = {"local": patch,
                   "global": ~patch}

    for area, subset in subsets.items():
        bs_tmp = bs_rate[subset]
        c_tmp = c_rate[subset]
        rate = np.append(bs_tmp, c_tmp, axis=1)

        fname = get_block_fname(baseline, conditional, is_patch, area=area)
        pca = PCA(rate.T, fname, n_components=components, force=force)

        bs_trans = pca.transform(bs_tmp.T).T
        c_trans = pca.transform(c_tmp.T).T

        title = f"{area.capitalize()} PCA of baseline and conditional data"
        ax = plot3D(c_trans, bs_trans, title=title)


def plot3D(condition:np.ndarray, baseline:np.ndarray, **kwargs):
    style = {"ls": "None",
             "marker": ".",}

    plt.figure(kwargs.get("num"))
    ax = plt.axes(projection="3d")
    ax.plot3D(*condition[:, ::], label="cond. data", color='b', **style)
    ax.plot3D(*baseline[:, ::], label="baseline data", color='r', **style)

    ax.set_xlabel("1 PC")
    ax.set_ylabel("2 PC")
    ax.set_zlabel("3 PC")
    ax.set_title(kwargs.get("title"))
    ax.legend()

    return ax



def get_block_fname(baseline:str, conditional:str, is_patch:bool=False, area:str=None):
    parts = [baseline, conditional]
    if is_patch:
        parts.append('patch')
        if area is not None:
            parts.append(area)
    fname = '_'.join(parts)

    return fname


def PCA(data:np.ndarray, fname:str, n_components:int=3, force:bool=False):
    try:
        pca = PIC.load(fname)
        if force:
            raise FileNotFoundError
    except FileNotFoundError:
        pca = sk.PCA(n_components=3)
        # n_samples x n_features
        pca.fit(data)
        PIC.save(fname, pca)
    return pca


def run_PCA(postfixes:list, force:bool=False):
    for s in postfixes:
        # n_samples x n_features
        try:
            pca = PIC.load(s)
            if force:
                raise FileNotFoundError
        except FileNotFoundError:
            pca = sk.PCA()
            rate = PIC.load_rate(postfix=s, skip_warmup=True, exc_only=True)
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


def hist_activity(rate_postfixes:list, rate_labels:list, delta_a:float=None):
    """
    Load the rates and makes a normalized histogram of it.

    Parameters
    ----------
    rate_postfixes : list
    rate_labels : list
        Displayed as legend.
    delta_a : float, optional
        Delta of the activity. The default is 0.1.

    Returns
    -------
    None.

    """
    delta_a = delta_a or 0.1

    rates = [PIC.load_rate(postfix, skip_warmup=True).flatten() for postfix in rate_postfixes]
    rates = np.asarray(rates)
    bins = np.arange(0, 1+delta_a, delta_a)

    weights = np.ones_like(rates) / rates.shape[1]
    plt.hist(rates.T, weights=weights.T, bins=bins)

    plt.legend(rate_labels)
    plt.xlabel("Activity")
    plt.ylabel("Percentage of occurence")


def number_of_sequences(neuron:(int, iter), avg:bool=False, postfix:str=None, threshold:float=None, min_dist:int=None)->int:
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
    threshold = threshold or 0.35
    min_dist = min_dist or CF.TAU

    rate = PIC.load_rate(postfix)
    rate = rate[:CF.NE, CF.WARMUP:]

    if isinstance(neuron, int):
        number =  number_of_peaks(rate[neuron], thres=threshold, min_dist=min_dist)
    else:
        if avg:
            number =  number_of_peaks(rate[neuron].mean(axis=0), thres=threshold, min_dist=min_dist)
        else:
            number = np.zeros(len(neuron))
            for idx, n in enumerate(neuron):
                number[idx] = number_of_peaks(rate[n], thres=threshold, min_dist=min_dist)

    return number


def number_of_peaks(data, **kwargs):
    no = putils.indexes(data, **kwargs, thres_abs=True).size
    return no




if __name__ == "__main__":
    analyze()
    # analyze_anatomy()
    # analyze_circular_dopamine_patch()
    # center = (1, 1)
    # radius = 2
    # p = DOP.circular_patch(CF.SPACE_WIDTH, center=center, radius=radius)
    # analyze_travel_direction(p, patchdetails=(center, radius))
    # center = (30, 18)
    # center = (20, 28)
    # radius = 2
    # p = DOP.circular_patch(CF.SPACE_WIDTH, center=center, radius=radius)
    # # plt.figure("with_linker_patch")
    # # plt.hist(number_of_sequences(p.nonzero()[0], avg=False,  postfix="linker"), )
    # # plt.hist(number_of_sequences(p, avg=True,  postfix="linker"), bins=1)
    # # plt.title(f"With DP patch\nNumber of sequences passed by @{center} (r={radius}) \nDetection as individual neurons or averaged activity.")
    # # plt.legend(["individual neurons", "mean"])

    # # plt.figure("with_linker_ACh_patch")
    # # plt.hist(number_of_sequences(p.nonzero()[0], avg=False,  postfix="linker_ACh"), )
    # # plt.hist(number_of_sequences(p, avg=True,  postfix="linker_ACh"), bins=1)
    # # plt.title(f"With ACh patch\nNumber of sequences passed by @{center} (r={radius}) \nDetection as individual neurons or averaged activity.")
    # # plt.legend(["individual neurons", "mean"])


    # def hist_sequences(rate_postfix:str, patch:np.ndarray, figname:str=None):
    #     plt.figure(figname)
    #     plt.hist(number_of_sequences(patch.nonzero()[0], avg=False,  postfix=rate_postfix))
    #     plt.hist(number_of_sequences(patch, avg=True,  postfix=rate_postfix), bins=1)
    #     plt.legend(["individual neurons", "mean"])


    # ana =  [("P2_bs", "baseline", "Baseline"),
    #         ("P2_25p", "dop25", "Dopamine patch (25%)"),
    #         ("P2_35p", "dop35", "Dopamine patch (35%)"),
    #         # ("g6_0_J2_5", "without_linker_patch", "Without DP patch"),
    #         # ("low_activity", "low_activity", "Lower J"),
    #         # ("repeater", "low_activity_with_patch", "Lower J with repeater"),
    #         ]

    # def template(header:str)->str:
    #     tmp = f"{header}\nNumber of sequences passed by @{center} (r={radius}) \nDetection as individual neurons or averaged activity."
    #     return tmp

    # for postfix, figname, header in ana:
    #     hist_sequences(rate_postfix=postfix, patch=p, figname=figname)
    #     plt.title(template(header))
    # pass
