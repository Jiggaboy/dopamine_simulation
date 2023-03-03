#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 20:54:57 2022

@author: hauke
"""

import cflogger
logger = cflogger.getLogger()

from collections.abc import Iterable
import matplotlib.pyplot as plt
import numpy as np

import lib.pickler as PIC
import universal as UNI

from plot.lib import plot_activity
from plot import activity
# from figure_generator.connectivity_distribution import set_layout

## Specifiy the Config here
from params import PerlinConfig, StarterConfig

BASELINE = True
PATCHES  = False

cfg = PerlinConfig()

def main():
    if BASELINE:
        baseline_average(cfg)
    if PATCHES:
        all_tags = cfg.get_all_tags()
        avg_activity(all_tags, cfg)



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
    fig = activity.activity(rates, norm=(0, .3), figname=figname, figsize=(3.6, 3))
    plt.title("Avg. activity")
    # set_layout(config.rows, margin=0, spine_width=1)
    PIC.save_figure(figname, fig, sub_directory=config.sub_dir)



def avg_activity(postfix:list, config)->None:
    postfix = UNI.make_iterable(postfix)

    for tag in postfix:
        logger.info(f"Load {tag}...")
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)

        plot_activity(avgRate, norm=(0, .5), figname=tag, figsize=(7, 6))
        plt.title("Avg. activity")
        #############
        # Make Details of the figure here!
        # set_layout(config.rows, margin=0)
        plt.savefig(PIC.get_fig_filename(tag + "_avg", format_="svg"), format="svg")
        plt.title((avgRate).mean())


def patchy_activity(activity:np.ndarray, patch:np.ndarray)->None:
    """
    activity, patch:
        2D array
    """
    plot_activity(activity[~patch], tag="patched_activity")



if __name__ == "__main__":
    main()
    plt.show()
