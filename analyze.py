#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021-03-13

@author: Hauke Wernecke

Analyses are:
    - Subspace analysis: Measures the principles angles between subspaces.
        Takes the Config and the raw_tags as input
        Hyperparameter are the radii of the analysis LOCAL_R and GLOBAL_R

    - Average rate: Averages the rate across time and saves the data.
        Visualization is separate

    - Joint PCA: For visualization purposes. Joints the data and performs a PCA.
        FORCE_PCA determines whether a pca object is loaded or a pca is performed.

"""
import cflogger
logger = cflogger.getLogger()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.decomposition as sk
import os

from peakutils import peak as putils

from custom_class.population import Population
import dopamine as DOP
from plot import angles as _plot_angles

import lib.pickler as PIC
import universal as UNI

import animation.activity as ACT
import animation.rate as RAT

from lib import SequenceCounter
from analysis.pca import PCA
from analysis.lib.subspace_angle import SubspaceAngle
from analysis import SequenceDetector


from params import BaseConfig, TestConfig, PerlinConfig, NullConfig, ScaleupConfig, StarterConfig, LowDriveConfig

### SELECT CONFIG
#Config = TestConfig()
Config = PerlinConfig()

################################ Average rate
AVERAGE_BASELINE_RATES = False
AVERAGE_RATES = False

# Details are parametrized in the config/analysisparams.
################################ DBSCAN sequences
RUN_DBSCAN = False


################################ passing sequences
DETECT_SEQUENCES = False
# RADIUS = 2
# MINIMAL_PEAK_DISTANCE = Config.TAU
RATE_THRESHOLD = 0.3
### Perlin Configuration size:4, base:1
SEQ_DETECTION_SPOTS = []


################################ Subspace analysis
RUN_SUBSPACE = True
ANGLE_PLOT = False
ANGLE_PLOT_PC = False
PATCH_CROSS_BASELINE = True
CROSS_ANGLES = False
CROSS_BASELINES = False
LOCAL_R = 8
GLOBAL_R = 24
ANGLE_RADIUS = (LOCAL_R,)# GLOBAL_R)#, None)

################################ Joint PCA
FORCE_PCA = False
radius_pca = 12
n_components = 3


def prepare_analysis():
    center = ((30, 18), (28, 26), )
    # UNI.append_spot(SEQ_DETECTION_SPOTS, "in", (center))
    # UNI.append_spot(SEQ_DETECTION_SPOTS, "edge", (center)
    # UNI.append_spot(SEQ_DETECTION_SPOTS, "out", (center)

    # UNI.append_spot(SEQ_DETECTION_SPOTS, "linker", ((21, 65), (67, 30), (30, 66)))
    # UNI.append_spot(SEQ_DETECTION_SPOTS, "repeater", ((9, 37), (2, 32), (55, 49)))
    # UNI.append_spot(SEQ_DETECTION_SPOTS, "repeater-proxy", ((2, 31), (29, 35), (29, 25)))

    center = ((67, 30), (50, 37), (60, 46))
    #center = ((35, 49), (49, 36), (29, 35), (29, 25))
    # UNI.append_spot(SEQ_DETECTION_SPOTS, "in-activator", (center))
    # UNI.append_spot(SEQ_DETECTION_SPOTS, "edge-activator", (center))
    UNI.append_spot(SEQ_DETECTION_SPOTS, "out-activator", (center))
    #UNI.append_spot(SEQ_DETECTION_SPOTS, "activator-proxy", (center))

    # UNI.append_spot(SEQ_DETECTION_SPOTS, "starter", ((46, 3), (49, 6), (43, 13)))
    # UNI.append_spot(SEQ_DETECTION_SPOTS, "starter", ((48, 8), ))


################################ TAGS ################################
# TODO: What to do with the tags? Keep them here? Split for each analysis?
TAGS = "starter", "out-activator"
TAGS = Config.center_range.keys()


# # PCA_ compare the manifolds

### Joint PCA: Just for visualization
# Requires the baseline and the conditional tag
# Parameters are also the radius and the center if considering a local patch.

def analyze():
    if AVERAGE_BASELINE_RATES:
        logger.info(f"Average rates: {Config.baseline_tags}")
        _average_rate(*Config.baseline_tags, sub_directory=Config.sub_dir, config=Config)
    if AVERAGE_RATES:
        tags = Config.get_all_tags(TAGS)
        logger.info(f"Average rates: {tags}")
        _average_rate(*tags, sub_directory=Config.sub_dir, config=Config)

    if RUN_DBSCAN:
        import analysis.dbscan_sequences as sequences_by_dbscan
        sequences_by_dbscan.analyze()

    if RUN_SUBSPACE:
        subspace_angle(Config, TAGS, plot_angles=ANGLE_PLOT, plot_PC=ANGLE_PLOT_PC)


    if DETECT_SEQUENCES:
        logger.info(f"Analyze spots: {SEQ_DETECTION_SPOTS}")
        detect_sequences(Config, SEQ_DETECTION_SPOTS)


    plt.show()
    return


#################################### PASSING SEQUENCES ########################################################################################

def detect_sequences(config:object, tag_spot:list):
    """
    Analyzes the # of sequences by peaks above threshold.
    Across seeds -> An object for each tag+seed (e.g. repeater_6_50_20_seed) in tag_spot is saved.
    Plot in plot.sequences.
    """
    for name, center in tag_spot:
        for seed in config.drive.seeds:
            tags = config.get_all_tags(name, seeds=seed)
            for tag in tags:
                counter = SequenceCounter(tag, center)

                counter.patch, counter.patch_avg = passing_sequences(center, config.analysis.sequence.radius, tag, config)
                counter.baseline, counter.baseline_avg = passing_sequences(center, config.analysis.sequence.radius, config.baseline_tag(seed), config)

                PIC.save_sequence(counter, counter.tag, sub_directory=config.sub_dir)


def passing_sequences(center, radius:float, tag:str, config:object):
    from analysis import SequenceDetector
    rate = PIC.load_rate(tag, exc_only=True, skip_warmup=True, sub_directory=config.sub_dir, config=config)

    sd = SequenceDetector(radius, RATE_THRESHOLD, MINIMAL_PEAK_DISTANCE)
    return sd.passing_sequences(rate, center, rows=config.rows)



#################################### DBSCAN #################################################################################################


#################################### SUBSPACE ANGLE #############################################################################################

def _plot_PC(config:object, *pcas:object, patch:np.ndarray, k:int=1, norm:tuple=None, figname:str=None):
    from animation import activity as plot_activity

    CMAP = plt.cm.seismic

    num = "PC"
    num = num if figname is None else num + figname + f"_{k}"

    norm = (-.5, .5)

    activity = []

    for pca in pcas:
        if patch is not None:
            patch_activity = np.zeros(config.rows**2)
            patch_2d = patch.reshape((config.rows, config.rows))
            patch_activity[patch] = pca.components_[k - 1]
        else:
            patch_activity = pca.components_[k - 1]
        activity.append(patch_activity)

    logger.info(f"Lenght of PCA data: {activity}")

    ax_titles = ["Baseline", "Patch"]
    title = f"Activation of the {k}-th PC ({num})"
    plot_activity.activity(*activity, figname=num, norm=norm, cmap=CMAP, figsize=(8, 6), title=title, ax_titles=ax_titles)

    plt.savefig(os.path.join("figures", "angle", num) + ".svg")



#################################### Average Rate #############################################################################################

def _average_rate(*tags, **save_params):
    """Averages the rates of the given tags. Saves the averaged rates."""
    for tag in tags:
        try:
            rate = PIC.load_rate(tag, exc_only=True, **save_params)
        except FileNotFoundError:
            logger.error(f"Averaging failed! Could not find file to the tag: {tag}")
            continue
        avgRate = rate.mean(axis=1)
        PIC.save_avg_rate(avgRate, tag, **save_params)







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

"""
def passing_sequences(center, radius, baseline:str, condition:str, figname:str="sequence", title:str=None):
    patch = DOP.circular_patch(CF.SPACE_WIDTH, center, radius)
    figname = f"{figname}_{baseline}"
    title = ""
    plot_passing_sequences(patch, postfix=baseline, figname=figname, title="Baseline simulation", details=(center, radius))
    figname = f"{figname}_{condition}"
    title = title or "Modulatory simulation"
    plot_passing_sequences(patch, postfix=condition, figname=figname, title=title, details=(center, radius))
"""

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


# To be tested: Changes are : plot_circle -> white_dashed_circle
def plot_patch(center:tuple, radius:int)->None:
    from plot.lib import white_dashed_circle
    white_dashed_circle(center, radius=radius)
    center = np.asarray(center)
    for idx, c in enumerate(center):
        if c + radius > CF.SPACE_WIDTH:
            n_center = center.copy()
            n_center[idx] = n_center[idx] - CF.SPACE_WIDTH
            white_dashed_circle(n_center, radius=radius)
    if all(center + radius > CF.SPACE_WIDTH):
        n_center = center.copy() - CF.SPACE_WIDTH
        white_dashed_circle(n_center, radius=radius)


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



def block_PCA(baseline:str, conditional:str, config, patch:np.ndarray=None, n_components:int=6, force:bool=False, plot_bs_first:bool=True, title:str=None):

    bs_rate = PIC.load_rate(postfix=baseline, skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)
    c_rate = PIC.load_rate(postfix=conditional, skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)


    if patch is None:
        is_patch = False
        patch = np.full(bs_rate.shape[0], fill_value=True)
        subsets = {"all": patch,}
    else:
        is_patch = True
        subsets = {
            "local": patch,
            # "all": np.full(bs_rate.shape[0], fill_value=True),
        }

    for area, subset in subsets.items():
        bs_tmp = bs_rate[subset]
        c_tmp = c_rate[subset]
        rate = np.append(bs_tmp, c_tmp, axis=1)

        fname = get_block_fname(baseline, conditional, is_patch, area=area)

        bs_trans = pca.transform(bs_tmp.T).T
        c_trans = pca.transform(c_tmp.T).T

        title = title or "Joint PCA of simulation w/ and w/o patch"
        title_a = f"{area.capitalize()}: {title}"
        ax = plot3D(c_trans, bs_trans, title=title_a, plot_bs_first=plot_bs_first, num=f"pca_{area}_{conditional}")

        print(f"Run ratio and return pcas of area: {area}")
    return
    # TODO
    return bs_pca, cond_pca


def plot3D(condition:np.ndarray, baseline:np.ndarray, **kwargs):
    style = {
        "ls": "dotted",
        "marker": ",",
        "linewidth": .6
    }

    C_BASELINE = "red"
    C_PATCH = "blue"

    plt.figure(kwargs.get("num"), figsize=(8, 8))
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
    plt.legend()


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



if __name__ == "__main__":
    prepare_analysis()
    analyze()
    plt.show()
