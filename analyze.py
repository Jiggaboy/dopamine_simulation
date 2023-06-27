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

# import animation.activity as ACT
# import animation.rate as RAT

from lib import SequenceCounter
from analysis.pca import PCA
from analysis.lib.subspace_angle import SubspaceAngle
from analysis import SequenceDetector


from params import BaseConfig, TestConfig, PerlinConfig, NullConfig, ScaleupConfig, StarterConfig, LowDriveConfig

### SELECT CONFIG
#Config = TestConfig()
Config = PerlinConfig()

################################ Average rate
AVERAGE_BASELINE_RATES = True
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
RUN_SUBSPACE = False
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
        from analysis import dbscan_sequences
        dbscan_sequences.analyze(Config)


    plt.show()
    return


#################################### DBSCAN #################################################################################################


#################################### SUBSPACE ANGLE #############################################################################################

def _plot_PC(config:object, *pcas:object, patch:np.ndarray, k:int=1, norm:tuple=None, figname:str=None):
    from plot import activity as plot_activity

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


#################################### PCA #############################################################################################




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



if __name__ == "__main__":
    analyze()
    plt.show()
