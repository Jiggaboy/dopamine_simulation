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
from plot import angles as plot_angles
    
import util.pickler as PIC

import animation.activity as ACT
import animation.rate as RAT


from params import BaseConfig, TestConfig, PerlinConfig
Config = TestConfig()
Config = PerlinConfig()
# Config = StarterConfig()

################################ Average rate
AVERAGE_RATE = True

################################ Subspace analysis
RUN_SUBSPACE = False
LOCAL_R = 12
GLOBAL_R = 24

################################ Joint PCA
FORCE_PCA = True
radius_pca = 12
n_components = 3

################################ DBSCAN of sequences
RUN_DBSCAN = True
EPS = 3
MIN_SAMPLES = 6
td = 1
SPIKE_THRESHOLD = 0.3



def _plot_cluster(data:np.ndarray, labels:np.ndarray=None, force_label:int=None):
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    ax.set_xlabel("time")
    ax.set_ylabel("X-Position")
    ax.set_zlabel("Y-Position")
    ax.set_ylim(0, 70)
    ax.set_zlim(0, 70)

    if labels is None:
        ax.scatter(*data.T, marker=".")
        return

    unique_labels = np.unique(labels)
    #ax.scatter(*data.T, marker=".")
    for l in unique_labels:
        if force_label is not None and l != force_label:
            continue
        ax.scatter(*data[labels == l].T, label=l, marker=".")
    plt.legend()


# # PCA_ compare the manifolds

#
raw_tags = "edge-activator", "linker"
raw_tags = "linker", "repeater"
raw_tags = "repeater", 
raw_tags = "edge-activator", "out-activator"
raw_tags = "in", "edge", "out"
raw_tags = "edge-activator", "linker"
raw_tags = "starter", "edge-activator"


### Joint PCA: Just for visualization
# Requires the baseline and the conditional tag
# Parameters are also the radius and the center if considering a local patch.

def analyze():
    if AVERAGE_RATE:
        all_tags = Config.get_all_tags()
        save_average_rate(*all_tags, sub_directory=Config.sub_dir, config=Config)
    
    if RUN_SUBSPACE:
        subspace_angle(Config, raw_tags)
        plt.show()
        
        
    if RUN_DBSCAN:
        dbscan(Config, tag=Config.baseline_tag)
    return
    
    
    for tag in raw_tags:
        print(f"run PCA for {tag}")
        center = Config.get_center(tag)
        patch = DOP.circular_patch(Config.rows, center, radius_pca)
        tags = Config.get_all_tags((tag,))
        for t in tags:
            block_PCA(Config.baseline_tag, t, config=Config, patch=patch, force=FORCE_PCA, n_components=n_components)

    plt.show()
    return
    pass


def dbscan(config:object, tag:str)->None:
    
    def load_coordinates_and_rate(cfg):
        from custom_class import Population
        pop = Population(cfg)
        rate = PIC.load_rate(cfg.baseline_tag, sub_directory=cfg.sub_dir, config=cfg, skip_warmup=True, exc_only=True)
        return pop.coordinates[:pop.exc_neurons.size], rate[:pop.exc_neurons.size][:, :1000]
    
    def binarize_rate(rate:np.ndarray, threshold:float=0.5):
        # Every activation above threshold is turned to a 1, 0 otherwise.
        rate[rate >= threshold] = 1
        rate[rate < threshold] = 0
        return rate

    coordinates, rate = load_coordinates_and_rate(Config)
    bin_rate = binarize_rate(rate.T, SPIKE_THRESHOLD)
    
    total_spikes = np.count_nonzero(bin_rate)
    spike_train = np.zeros((3, total_spikes), dtype=int)
    
    start = end = 0
    for t in range(bin_rate.shape[0]):
        S_t = bin_rate[t, :].nonzero()[0]
        spikes = S_t.size

        end += spikes
        spike_train[:, start:end] = np.vstack([np.full(fill_value=t / td, shape=spikes), coordinates[S_t].T])
        start += spikes
        
    from .dbscan import DBScan
    db = DBScan(eps=EPS, min_samples=MIN_SAMPLES)
    data, labels = db.fit_toroidal(spike_train.T, nrows=Config.rows)
    print(data.shape)
    print(data[labels == 0].shape)
    SUBSAMPLE = 8
    _plot_cluster(data[::SUBSAMPLE], labels[::SUBSAMPLE], force_label=None)
    plt.show()

def subspace_angle(config:object, plain_tags:list, plot:bool=True, plot_PC:bool=True)->None:
    from .subspace_angle import SubspaceAngle
    angle = SubspaceAngle(Config)
    
    for r_tag in plain_tags:
        tags = config.find_tags((r_tag,))
        print(f"Tags are: {tags}")
        for tag in tags:
            center = Config.get_center(r_tag)
            for r in (LOCAL_R, GLOBAL_R, None):
                try:
                    mask = DOP.circular_patch(config.rows, center=center, radius=r)
                except TypeError:
                    print("TypeError: No masked used!")
                    mask = None
                angle.fit(tag, mask=mask)
                t = tag + str(r)
                if plot:
                    plot_angles.cumsum_variance(angle, tag=t)
                    plot_angles.angles(angle, tag=t)
                if plot_PC:
                    _plot_PC(config, angle.pcas[0], mask, figname=f"bs_{tag}_{r}")
                    _plot_PC(config, angle.pcas[1], mask, figname=f"{tag}_{r}")
        


def _plot_PC(config, pca, patch:np.ndarray, k:int=1, norm:tuple=None, figname:str=None):
    from plot.lib import plot_activity

    CMAP = plt.cm.seismic

    num = "PC"
    num = num if figname is None else num + figname

    norm = (-.5, .5)

    if patch is not None:
        patch_activity = np.zeros(config.rows**2)
        patch_2d = patch.reshape((config.rows, config.rows))
        patch_activity[patch] = pca.components_[k - 1]
    else:
        patch_activity = pca.components_[k - 1]
    
    plot_activity(patch_activity, figname=num, norm=norm, cmap=CMAP, figsize=(8, 6), title=f"Activation of the k-th PC ({num})")
    
    plt.savefig(os.path.join("figures", "angle", num) + ".svg")

    
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
        pca = PCA(rate.T, fname, n_components=n_components, force=force)

        bs_trans = pca.transform(bs_tmp.T).T
        c_trans = pca.transform(c_tmp.T).T

        title = title or "Joint PCA of simulation w/ and w/o patch"
        title_a = f"{area.capitalize()}: {title}"
        ax = plot3D(c_trans, bs_trans, title=title_a, plot_bs_first=plot_bs_first, num=f"pca_{area}_{conditional}")

        print(f"Run ratio and return pcas of area: {area}")
    return 
    # TODO
    return bs_pca, cond_pca


def ratio_PCA(data, n_components:int=50, tags:tuple=("Data", ), plot:bool=True):
    pca = PCA(data, None, n_components=n_components, force=True)
    cumsumVariances = sum_variances(pca.explained_variance_ratio_)
    if plot:
        plot_explained_variance_ratio(cumsumVariances, " - ".join(tags))
    return pca


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


def PCA(data:np.ndarray, fname:str, n_components:int=3, force:bool=False):
    try:
        pca = PIC.load(fname)
        if force:
            raise FileNotFoundError
    except (FileNotFoundError, TypeError):
        pca = sk.PCA(n_components=n_components, svd_solver='full')
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
    plt.legend()


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
