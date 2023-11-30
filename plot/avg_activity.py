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
import lib.universal as UNI

from plot.lib import plot_activity
from plot.activity_difference import plot_patch_from_tag
from plot import activity

## Specifiy the Config here
from params import PerlinConfig

# These parameter are used if run as main()
BASELINE_AVERAGE = True
BASELINE_SEEDS = True
PATCHES_SEEDS  = True
standalone_config = PerlinConfig()



#===============================================================================
# METHODS
#===============================================================================
def plot_avg_activity(config:object=standalone_config, plot_baseline_average:bool=BASELINE_AVERAGE, baseline_seeds:bool=BASELINE_SEEDS, patches_seeds:bool=PATCHES_SEEDS):
    if plot_baseline_average:
        baseline_average(config)
    if baseline_seeds:
        all_tags = config.baseline_tags
        avg_activity(all_tags, config)
    if patches_seeds:
        all_tags = config.get_all_tags()
        avg_activity(all_tags, config)
    plt.show()



def baseline_average(config:object):
    tags = config.baseline_tags

    rates = []
    for tag in tags:
        logger.info(f"Load {tag}...")
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)
        rates.append(avgRate)
    rates = np.asarray(rates)

    if rates.ndim > 1:
        rates = rates.mean(axis=0)

    figname = "baseline_averaged_across_seeds"
    fig = activity.activity(rates, norm=(0, .5), figname=figname, figsize=(3.6, 3))
    plt.title("Avg. activity")
    PIC.save_figure(figname, fig, sub_directory=config.sub_dir)



def avg_activity(postfix:list, config)->None:
    postfix = UNI.make_iterable(postfix)

    for tag in postfix:
        logger.info(f"Load {tag}...")
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)

        plot_activity(avgRate, norm=(0, .5), figname=tag, figsize=(7, 6))

        try:
            plot_patch_from_tag(tag, config)
        except KeyError:
            logger.info(f"Could not find patch for tag: {tag}")
        plt.title("Avg. activity")

        plt.savefig(PIC.get_fig_filename(tag + "_avg", format_="svg"), format="svg")
        plt.title((avgRate).mean())


def patchy_activity(activity:np.ndarray, patch:np.ndarray)->None:
    """
    activity, patch:
        2D array
    """
    plot_activity(activity[~patch], tag="patched_activity")



if __name__ == "__main__":
    plot_avg_activity(config=standalone_config)
    plt.show()
