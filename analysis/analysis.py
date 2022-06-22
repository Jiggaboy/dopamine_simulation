#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:38:43 2021

@author: hauke
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.decomposition as sk

from peakutils import peak as putils

from custom_class.population import Population
import configuration as CF
import dopamine as DOP

import util.pickler as PIC

import animation.activity as ACT
import animation.rate as RAT


from params import BaseConfig, TestConfig, PerlinConfig, StarterConfig
Config = TestConfig()
Config = PerlinConfig()
Config = StarterConfig()


def analyze():
    all_tags = Config.get_all_tags()
    all_tags.append(Config.baseline_tag)
    save_average_rate(*all_tags, sub_directory=Config.sub_dir, config=Config)


    # # PCA_ compare the manifolds
    # force = False
    # n_components = 3
    #
    # radius_pca = 8
    #
    # tags = "starter",
    # centers = (24, 64), (63, 35)
    #
    # for tag, center in zip(tags, centers):
    #     # center = Config.center_range[tag]
    #     patch = DOP.circular_patch(Config.rows, center, radius_pca)
    #     linker_tag = [t for t in all_tags if t.startswith(tag)][0]
    #     block_PCA(Config.baseline_tag, linker_tag, config=Config, patch=patch, force=force, n_components=n_components)
    #
    # return
    # center = Config.center_range["linker"]
    # patch = DOP.circular_patch(Config.rows, center, radius_pca)
    # linker_tag = [t for t in all_tags if t.startswith("linker")][0]
    # block_PCA(Config.baseline_tag, linker_tag, patch=patch, force=force, n_components=n_components)
    #
    # center = Config.center_range["edge-activator"]
    # patch = DOP.circular_patch(Config.rows, center, radius_pca)
    # activator_tag = [t for t in all_tags if t.startswith("edge-activator")][0]
    # block_PCA(Config.baseline_tag, activator_tag, patch=patch, force=force, n_components=n_components)

    # return

    # # 1 Making a histogram of the rates of the neurons.
    # # rate = ["baseline", "dop", "out-degree"]
    # # rate_labels = ["baseline", "in-degree", "out-degree"]
    # # hist_activity(rate, rate_labels)

    # # 41_55
    # # Compare multiple setups using the avg. activity
    # rate_postfixes = [
    #     # "41_55_link", "41_55_link_ach",
    #     # "41_55_repeat", "41_55_repeat_ach",
    #     # "41_55_start",
    #     # "41_55_in",
    #     # "41_55_in_2",
    #     # "41_55_edge",
    #     # "41_55_edge_2",
    #     # "41_55_out",
    #     # "41_55_out_2",
    #     # "41_55_rand",
    #     # "41_55_baseline",
    #     "linker_6_50_1.0_20",
    #     "Perlin_uniform_baseline",
    #     ]
    # for t in all_tags:
    #     avgRate = PIC.load_average_rate(t, sub_directory=Config.sub_dir, config=Config)
    #     plt.figure()
    #     plt.imshow(avgRate.reshape(Config.rows, Config.rows))
    # quit()
    # # in/edge/out patch
    # center = (35, 26)
    # # analyze_circular_dopamine_patch(rate_postfixes, plot=True, center=center)
    # # starter
    # # center = (43, -2)
    # # center = (43, 68)
    # # analyze_circular_dopamine_patch(rate_postfixes, plot=True, center=center)
    # # repeater
    # # center = (17, 34)
    # # analyze_circular_dopamine_patch(rate_postfixes, plot=True, center=center, title="ACh-patch")
    # # linker
    # # center = (16, 56)
    # # analyze_circular_dopamine_patch(rate_postfixes, plot=True, center=center, title="ACh-patch")
    # # activator
    # # center = (63, 34)
    # # analyze_circular_dopamine_patch(rate_postfixes, plot=True, center=center, title="DP-patch")
    # # control
    # # analyze_circular_dopamine_patch(rate_postfixes, plot=True)

    # merge_avg_rate_to_key(rate_postfixes, plot=True)

    # center_post, radius = (35, 18), 2
    # # control: in pos.
    # center_pre, radius = (66, 34), 2
    # # passing_sequences_pre_post(center_pre, center_post, radius, "41_55_baseline", "41_55_rand", title="Random patch")
    # return

    # #-----------------------------------------------------------------------------

    # MODE = "Perlin_uniform"
    # center_range = (
    #     (17, 34), # repeater
    #     (43, 68), # starter
    #     (16, 56), # linker
    #     (66, 34), # in-activator
    #     (63, 34), # edge-activator
    #     (59, 34), # out-activator
    #     (35, 18), # in
    #     (35, 22), # edge
    #     (35, 26), # out
    # )

    # RADIUSES = (6, 12, 18)
    # AMOUNT_NEURONS = (10, 50, 100)
    # PERCENTAGES = (.3, .2, .1)

    # baseline_postfix = "_".join((MODE, "baseline"))

    # neuron_population = Population.load(CF.SPACE_WIDTH, mode=MODE)
    # centers= (
    #     (56, 47), # main far
    #     (56, 49),
    #     (54, 51),
    #     (50, 52),
    #     (31, 54), # main close
    #     (30, 55),
    #     (28, 55),
    #     (21, 49),
    #     (16, 56), # linker
    #     (16, 57),
    #     (21, 65),
    #     (20, 68),
    #   )
    # # center_peaks = []

    # baserate = PIC.load_rate(baseline_postfix, skip_warmup=True, exc_only=True)

    # delta_t = 250 #ms


    # rates = np.zeros(shape=(len(centers), baserate.shape[1]))

    # for i, center in enumerate(centers):
    #     idx = DOP.center_to_idx(neuron_population.coordinates, center)
    #     rate = baserate[idx]
    #     rates[i] = rate
    # ccoef = np.corrcoef(rates)
    # plot_corrcoef(ccoef)
    # plt.title("Baseline - no shift")

    # # 1. is reference
    # n_rates = np.zeros(shape=(len(centers), baserate.shape[1] - 2*delta_t))
    # for i, rate in enumerate(rates[:]):
    #     corr = np.correlate(rates[0], rate, mode="full")
    #     shift = (len(rates[0]) - 1) - np.argmax(corr)
    #     if abs(shift) >= delta_t:
    #         shift = 0
    #     n_rates[i] = rate[delta_t+shift:-delta_t+shift]

    # ccoef = np.corrcoef(n_rates)
    # plot_corrcoef(ccoef)
    # plt.title("Baseline - shift")


    # rates = np.zeros(shape=(len(centers), baserate.shape[1]))
    # repeater = "Perlin_uniform_16_56_6_50_20"
    # for i, center in enumerate(centers):
    #     idx = DOP.center_to_idx(neuron_population.coordinates, center)
    #     rate = PIC.load_rate(repeater, skip_warmup=True, exc_only=True)[idx]
    #     rates[i] = rate

    # ccoef = np.corrcoef(rates)
    # plot_corrcoef(ccoef)
    # plt.title("Linker - no shift")

    # # 1. is reference
    # n_rates = np.zeros(shape=(len(centers), baserate.shape[1] - 2*delta_t))
    # for i, rate in enumerate(rates[:]):
    #     corr = np.correlate(rates[0], rate, mode="full")
    #     shift = (len(rates[0]) - 1) - np.argmax(corr)

    #     if abs(shift) >= delta_t:
    #         shift = 0
    #     n_rates[i] = rate[delta_t+shift:-delta_t+shift]

    # ccoef = np.corrcoef(n_rates)
    # plot_corrcoef(ccoef)
    # plt.title("Linker - shift")

    # # t = 50
    # # for p in center_peaks[0]:
    # #     test = center_peaks[1] > p - 50
    # #     test2 = center_peaks[1] < p + 50
    # #     joined = np.where(test & test2)[0][0]
    # #     print(joined)

    #     # print(np.any(p - t < center_peaks[1].any( < p + t))



    # rate_postfixes = []
    # for radius in RADIUSES[:1]:
    #     for center in center_range[2:3]:
    #         for amount in AMOUNT_NEURONS[1:2]:
    #             for percent in PERCENTAGES[1:2]:
    #                 tag = (MODE, *center, radius, amount, int(percent*100))
    #                 log = "_".join((str(e) for e in tag))
    #                 print(log)
    #                 rate_postfixes.append(log)
    #                 print(PIC.load_rate(log, skip_warmup=True, exc_only=True).mean())
    #                 # print(PIC.load_rate(log, skip_warmup=True, exc_only=True).shape)
    #                 # print(PIC.load_rate(f"{MODE}_warmup").shape)

    #                 plot_rates_vs_baseline([log], baseline=baseline_postfix, plot=True, center=center, radius=radius,  title=f"Linker - {amount} neurons - {percent*100}%")
    # #             # analyze_circular_dopamine_patch(rate_postfixes, plot=True, center=(69, 68), title="Repeater")
    # #             # analyze_circular_dopamine_patch(rate_postfixes, plot=True, center=center, title="Repeater")


    #-----------------------------------------------------------------------------

    # Sequence passings
    # IN
    # center, radius = (30, 18), 2
    # passing_sequences(center, radius, "41_55_baseline", "41_55_in_2", figname="in_2")
    # passing_sequences(center, radius, "41_55_baseline", "41_55_edge_2", figname="edge_2")
    # passing_sequences(center, radius, "41_55_baseline", "41_55_out_2", figname="out_2")
    # Starter
    # center, radius = (47, 3), 2
    # passing_sequences(center, radius, "41_55_baseline", "41_55_start", figname="starter")
    # repeater
    # center_post, radius = (2, 31), 2
    # center_pre, radius = (29, 35), 2
    # passing_sequences_pre_post(center_pre, center_post, radius, "41_55_baseline", "41_55_repeat", title="Repeater DP-patch")
    # passing_sequences_pre_post(center_pre, center_post, radius, "41_55_baseline", "41_55_repeat_ach", title="Repeater ACh-patch")
    # linker
    # center_post, radius = (21, 65), 2
    # linker: neighbouring sequence
    # center_pre, radius = (30, 61), 2
    # passing_sequences(center, radius, "41_55_baseline", "41_55_link", figname="linker_adj")
    # passing_sequences(center, radius, "41_55_baseline", "41_55_link_ach", figname="linker_ach_adj")
    # passing_sequences_pre_post(center_post, center_pre, radius, "41_55_baseline", "41_55_link", title="Linker DP-patch")
    # passing_sequences_pre_post(center_post, center_pre, radius, "41_55_baseline", "41_55_link_ach", title="Linker ACh-patch")
    # activator
    # center_post, radius = (35, 49), 2
    # # activator: main sequence
    # center_pre, radius = (49, 36), 2
    # passing_sequences_pre_post(center_post, center_pre, radius, "41_55_baseline", "41_55_edge", title="Activator DP-patch")
    # control: in-2-position
    # center_post, radius = (35, 18), 2
    # control: in pos.
    # center_pre, radius = (66, 34), 2
    # passing_sequences_pre_post(center_pre, center_post, radius, "41_55_baseline", "41_55_rand", title="Random patch")


    # run_PCA(rate_postfixes)

    # # PCA_ compare the manifolds
    # force = False
    # n_components = 3

    # # center, radius = (24, 64), 8 # link: similar
    # patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
    # block_PCA("41_55_baseline", "41_55_link", patch=patch, force=force, n_components=n_components, title="DP-Linker")
    # # block_PCA("41_55_baseline", "41_55_link_ach", patch=patch, force=force, n_components=n_components, plot_bs_first=False, title="ACh-Linker")
    # center, radius = (63, 35), 8 # edge: huge difference
    # patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
    # block_PCA("41_55_baseline", "41_55_edge", patch=patch, force=force, n_components=n_components, title="DP-Activator")
    # center, radius = (17, 34), 8 # repeater: more space for dop., less for ach
    # patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
    # block_PCA("41_55_baseline", "41_55_repeat", patch=patch, force=force, n_components=n_components, title="DP-Repeater")
    # block_PCA("41_55_baseline", "41_55_repeat_ach", patch=patch, force=force, plot_bs_first=False, n_components=n_components, title="ACh-Repeater")
    # CONTROL
    # center, radius = (17, 34), 8 # repeater: more space for dop., less for ach
    # patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
    # block_PCA("41_55_baseline", "41_55_repeat", patch=patch, force=force, n_components=n_components, title="DP-Repeater")
    # block_PCA("41_55_baseline", "41_55_repeat_ach", patch=patch, force=force, plot_bs_first=False, n_components=n_components, title="ACh-Repeater")



    # VELOCITY
    # radius 6, s:4, vs:2
    # center, radius = (1, 18), 4
    # patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
    # snapshots = ["v_dop25vs", "v_base", "v_dop25s"]
    # for s in snapshots:
    #     analyze_travel_direction(patch, (center, radius), postfix=s,
    #                           delta_t=40, threshold=0.2, plot_rates=False)


    # analyze_circular_dopamine_patch(rate_postfixes)


    pass

def plot_corrcoef(corrcoef):
    norm = (-1, 1)
    plt.figure()
    plt.imshow(corrcoef, cmap=plt.cm.seismic, vmin=norm[0], vmax=norm[1])
    plt.colorbar()


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


def passing_sequences_pre_post(center_pre, center_post, radius, baseline:str, condition:str, figname:str="sequence", title:str=None):
    patch_pre = DOP.circular_patch(CF.SPACE_WIDTH, center_pre, radius)
    patch_post = DOP.circular_patch(CF.SPACE_WIDTH, center_post, radius)
    patches = [patch_pre, patch_post]
    figname = f"{figname}_{baseline}"
    plot_passing_sequences_pre_post(patches, postfix=baseline, figname=figname, title="Baseline simulation")
    figname = f"{figname}_{condition}"
    plot_passing_sequences_pre_post(patches, postfix=condition, figname=figname, title=title)


def passing_sequences(center, radius, baseline:str, condition:str, figname:str="sequence", title:str=None):
    patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
    figname = f"{figname}_{baseline}"
    title = ""
    plot_passing_sequences(patch, postfix=baseline, figname=figname, title="Baseline simulation", details=(center, radius))
    figname = f"{figname}_{condition}"
    title = title or "Modulatory simulation"
    plot_passing_sequences(patch, postfix=condition, figname=figname, title=title, details=(center, radius))


def plot_passing_sequences(patch:np.ndarray, postfix:str, figname:str, title:str=None, details:tuple=None, details_in_title:bool=True):
    plt.figure(figname, figsize=(4, 3.2))
    pos = "" if details is None else f" @{details[0]} with r={details[1]}"
    plt.hist(number_of_sequences(patch.nonzero()[0], avg=False,  postfix=postfix), label=f"Individual neurons")
    passed_sequences = number_of_sequences(patch, avg=True,  postfix=postfix)
    width = .3
    plt.hist(passed_sequences, bins=[passed_sequences-width, passed_sequences+width], weights=[1], label=f"Avg. activity", )
    plt.title(f"{title}\nSequence passed: {passed_sequences}{pos}")
    plt.legend()


def plot_rates_vs_baseline(postfixes:list, baseline:str=None, **kwargs):
    rates = merge_avg_rate_to_key(postfixes, **kwargs)
    if baseline is not None:
        bs = PIC.load_rate(baseline, skip_warmup=True, exc_only=True)
        baseline_rate = bs.mean(axis=1)
        for rate in rates.items():
            plot_rate_difference(rate, baseline_rate, norm=(-.3, .3))


# TODO :Überflüssig?
def analyze_circular_dopamine_patch(postfixes:list, **kwargs):
    rates = merge_avg_rate_to_key(postfixes, **kwargs)
    plot_rate_differences(rates, norm=(-.3, .3))


def merge_avg_rate_to_key(keys:list, plot:bool=False, center:tuple=None, radius:float=4, title:str=None, config=None)->dict:
    rates = {}
    for s in keys:
        rate = PIC.load_rate(s, skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)
        # rate = PIC.load_rate(s, skip_warmup=True, exc_only=True)
        avgRate = rate.mean(axis=1)
        rates[s] = avgRate
        if plot:
            title= title or "Activity averaged across time"
            ACT.activity(avgRate, title=title, figname=f"circ_patch_{s}", norm=(0, 0.5))
            if center is not None:
                plot_patch(center, radius)
    return rates


def plot_patch(center:tuple, radius:int)->None:
    plot_circle(center, radius=radius)
    center = np.asarray(center)
    for idx, c in enumerate(center):
        if c + radius > CF.SPACE_WIDTH:
            n_center = center.copy()
            n_center[idx] = n_center[idx] - CF.SPACE_WIDTH
            plot_circle(n_center, radius=radius)
    if all(center + radius > CF.SPACE_WIDTH):
        n_center = center.copy() - CF.SPACE_WIDTH
        plot_circle(n_center, radius=radius)



def plot_circle(center, radius):
    circle = mpatches.Circle(center, radius=radius, fc="None", ec="white", linewidth=2, ls="dashed")
    plt.gca().add_artist(circle)


def plot_rate_difference(avg_rate:(str, np.ndarray), baseline:np.ndarray, norm:tuple=None):
    norm = norm or (None, None)

    rate_diff = avg_rate[1] - baseline
    diff_percent = rate_diff.mean() / baseline.mean()
    # To be adjusted
    figname = f"circ_patch_{avg_rate[0]}_bs"
    title = f"Network changes: \nActivation difference: {100 * diff_percent:+.2f}%"
    ACT.activity(rate_diff, figname=figname, title=title, norm=norm, cmap=plt.cm.seismic)


def plot_rate_differences(avg_rates:dict, norm:tuple=None):
    norm = norm or (None, None)

    high = []
    for key_i, rate_i in avg_rates.items():
        high.append(key_i)
        for key_j, rate_j in avg_rates.items():
            if key_j in high:
                continue
            rate_diff = rate_i - rate_j
            diff_percent = rate_diff.mean() / rate_j.mean()
            # To be adjusted
            figname = f"circ_patch_{key_i}_{key_j}"
            title = f"{key_i} - {key_j}: {rate_diff.mean():.5f}"

            title = f"Network changes: Random selection\nActivation difference: {100 * diff_percent:+.2f}%"
            ACT.activity(rate_diff, figname=figname, title=title, norm=norm, cmap=plt.cm.seismic)


def get_peaks(neuron_idx:int, postfix:str, delta_t:float=None, threshold:float=None):
    delta_t = delta_t or 50
    threshold = threshold or 0.35

    # load rate
    rate = PIC.load_rate(postfix, skip_warmup=True, exc_only=True)
    rate = rate[neuron_idx]

    peaks = putils.indexes(rate, thres=threshold, min_dist=delta_t, thres_abs=True)
    peaks = peaks[peaks > delta_t]
    peaks = peaks[peaks + delta_t < rate.size]
    return peaks




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
    title_pre =title + f"\n No. threshold crossings: {crossings.size}"
    title_post = title + f"\n Delta t: {delta_t}ms"
    des =  {"title_pre": title_pre,
            "title_post": title_post,}

    ACT.pre_post_activity(snapshot_pre, snapshot_post, **des)



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


def block_PCA(baseline:str, conditional:str, config, patch:np.ndarray=None, n_components:int=6, force:bool=False, plot_bs_first:bool=True, title:str=None):

    bs_rate = PIC.load_rate(postfix=baseline, skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)
    c_rate = PIC.load_rate(postfix=conditional, skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)

    if patch is None:
        is_patch = False
        patch = np.full(bs_rate.shape[0], fill_value=True)
        subsets = {"global": patch,}
    else:
        is_patch = True
        subsets = {
            "local": patch,
            # "global": ~patch
        }

    for area, subset in subsets.items():
        bs_tmp = bs_rate[subset]
        c_tmp = c_rate[subset]
        rate = np.append(bs_tmp, c_tmp, axis=1)

        fname = get_block_fname(baseline, conditional, is_patch, area=area)
        pca = PCA(rate.T, fname, n_components=n_components, force=force)

        bs_trans = pca.transform(bs_tmp.T).T
        c_trans = pca.transform(c_tmp.T).T

        title = title or "Joint PCA of simulation w/ and w/o patch"
        title_a = f"{area.capitalize()}: {title}"
        ax = plot3D(c_trans, bs_trans, title=title_a, plot_bs_first=plot_bs_first, num=f"pca_{area}_{conditional}")

        # ratio_PCA(bs_tmp.T, n_components, tags=(area, "Baseline"))
        # ratio_PCA(c_tmp.T, n_components, tags=(area, "Condition"))
    # plt.legend()


def ratio_PCA(data, n_components:int=50, tags:tuple=("Data", )):
    pca = PCA(data, None, n_components=n_components, force=True)
    cumsumVariances = sum_variances(pca.explained_variance_ratio_)
    plot_explained_variance_ratio(cumsumVariances, " - ".join(tags))


def plot3D(condition:np.ndarray, baseline:np.ndarray, **kwargs):
    style = {
        "ls": "dotted",
        "marker": ",",
        "linewidth": .6
    }

    C_BASELINE = "red"
    C_PATCH = "blue"


    plt.figure(kwargs.get("num"), figsize=(8, 8))
    # plt.figure(kwargs.get("num"), figsize=(3.4, 3))
    ax = plt.axes(projection="3d")
    bs_zorder = kwargs.get("plot_bs_first", True)
    bs_zorder = 2 if bs_zorder else 0
    ax.plot3D(*baseline[:3], color=C_BASELINE, **style, zorder=bs_zorder)
    ax.plot3D(*condition[:3], color=C_PATCH, **style, zorder=1)

    ax.set_xlabel("1 PC")
    ax.set_ylabel("2 PC")
    ax.set_zlabel("3 PC")
    ax.set_title(kwargs.get("title"))
    baseline = mpatches.Patch(color = C_BASELINE, label="Baseline")
    patch = mpatches.Patch(color = C_PATCH, label="Patch")
    ax.legend(handles=[baseline, patch])
    # plt.savefig()

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
    except (FileNotFoundError, TypeError):
        pca = sk.PCA(n_components=n_components)
        # n_samples x n_features
        pca.fit(data)
        if fname is not None:
            PIC.save(fname, pca)
    return pca


def run_PCA(postfixes:list, force:bool=False):
    for s in postfixes:
        rate = PIC.load_rate(postfix=s, skip_warmup=True, exc_only=True)
        pca = PCA(rate.T, fname=s, n_components=100, force=force)
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

    rates = [PIC.load_rate(postfix, skip_warmup=True, exc_only=True).flatten() for postfix in rate_postfixes]
    rates = np.asarray(rates)
    bins = np.arange(0, 1+delta_a, delta_a)

    weights = np.ones_like(rates) / rates.shape[1]
    plt.hist(rates.T, weights=weights.T, bins=bins)

    plt.legend(rate_labels)
    plt.xlabel("Activity")
    plt.ylabel("Percentage of occurence")




def save_average_rate(*tags, **save_params):
    for t in tags:
        rate = PIC.load_rate(t, skip_warmup=True, exc_only=True, **save_params)
        avgRate = rate.mean(axis=1)
        PIC.save_avg_rate(avgRate, t, **save_params)


# def load_average_rate(tag, sub_directory:str=None):
#     return PIC.load_rate("avg_" + tag, sub_directory=sub_directory)


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
