#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
import cflogger
logger = cflogger.getLogger()

import matplotlib.pyplot as plt
import numpy as np

import lib.pickler as PIC
from lib.pickler_class import Pickler
import lib.universal as UNI

from plot.activity_difference import plot_patch_from_tag
from plot.lib.frame import create_image
from plot.lib.basic import add_colorbar
from plot import activity
from plot.constants import COLOR_MAP_ACTIVITY, NORM_ACTIVITY, COLOR_MAP_DIFFERENCE

## Specifiy the Config here
from params import config

# These parameter are used if run as main()
BASELINE_AVERAGE = True
BASELINE_SEEDS = False
PATCHES_SEEDS  = True



#===============================================================================
# METHODS
#===============================================================================
def plot_avg_activity(config:object, plot_baseline_average:bool=BASELINE_AVERAGE, baseline_seeds:bool=BASELINE_SEEDS, patches_seeds:bool=PATCHES_SEEDS):
    # Average activity of all baseline simulations (averaged)
    if plot_baseline_average:
        baseline_average(config)
    # Average activity of a baseline simulation (individual runs)
    if baseline_seeds:
        all_tags = config.baseline_tags
        avg_activity(all_tags, config)
    # Average activity of the patches (Similar figures to the baseline ones)
    if patches_seeds:
        all_tags = config.get_all_tags()
        avg_activity(all_tags, config)



def baseline_average(config:object):
    tags = config.baseline_tags

    # Gather all rates
    rates = []
    for tag in tags:
        logger.info(f"Load {tag}...")
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)
        rates.append(avgRate)
    rates = np.asarray(rates)

    # Average if more than one run
    if rates.ndim > 1:
        rates = rates.mean(axis=0)


    figname = "baseline_averaged_across_seeds"

    fig, ax = plt.subplots(num=figname, figsize=(4, 3.4))
    norm = (0, 0.5)
    cmap = COLOR_MAP_ACTIVITY

    create_image(rates, norm, cmap, axis=ax)
    add_colorbar(ax, norm, cmap)

    plt.title("Avg. activity across seeds")
    plt.xlabel("X-Position")
    plt.ylabel("Y-Position")
    plt.xticks([10, 40, 70])
    plt.yticks([10, 40, 70])
    plt.tight_layout()

    pickler = Pickler(config)
    pickler.save_figure(figname, fig)



def avg_activity(postfix:list, config:object) -> None:
    postfix = UNI.make_iterable(postfix)

    pickler = Pickler(config)

    for tag in postfix:
        logger.info(f"Load {tag}...")
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)


        fig = plt.figure(tag, figsize=(7, 6))
        create_image(avgRate, norm=NORM_ACTIVITY, cmap=COLOR_MAP_ACTIVITY)
        plt.title("Avg. activity")
        plt.colorbar()

        try:
            plot_patch_from_tag(tag, config)
        except KeyError:
            logger.info(f"Could not find patch for tag: {tag}")

        pickler.save_figure(tag, fig)
        plt.title((avgRate).mean())


if __name__ == "__main__":
    plot_avg_activity(config)
    plt.show()
