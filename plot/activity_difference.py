#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 23:11:21 2022

@author: hauke
"""


from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import util.pickler as PIC
import universal as UNI

from plot.lib import plot_activity

## Specifiy the Config here
from params import PerlinConfig

NORM = (-.3, .3)


def main():
    cf = PerlinConfig()
    all_tags = cf.get_all_tags()
    all_tags = [t for t in all_tags if t.startswith("edge-activator")]
    activity_difference(cf.baseline_tag, all_tags, cf)

def activity_difference(baseline:str, postfixes:list, config, **kwargs):
    baseRate = PIC.load_average_rate(baseline, sub_directory=config.sub_dir, config=config)
    for tag in postfixes:
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)

        figname = f"patch_{tag}_bs"
        # title = f"Network changes: \nActivation difference: {100 * diff_percent:+.2f}%"
        # ACT.activity(rate_diff, figname=figname, title=title, norm=norm, cmap=plt.cm.seismic)
        plot_activity(avgRate - baseRate, figname=figname, norm=NORM, cmap=plt.cm.seismic, figsize=(10, 8))

        # Plot the patch
        name = tag.split("_")[0]
        center = config.center_range[name]

        radius = tag.split("_")[1]
        plot_patch(center, int(radius), width=config.rows)
        #############
        # Make Details of the figure here!
        from figure_generator.connectivity_distribution import set_layout
        set_layout(70, margin=0)
        plt.savefig(UNI.get_fig_filename(figname, format_="svg"), format="svg")
        plt.title((avgRate - baseRate).mean())



def plot_patch(center:tuple, radius:int, width:int)->None:
    # Plot the circle on location
    plot_circle(center, radius=radius)

    # Plot the circle on the other side of the toroid
    center = np.asarray(center)
    for idx, c in enumerate(center):
        if c + radius > width:
            n_center = center.copy()
            n_center[idx] = n_center[idx] - width
            plot_circle(n_center, radius=radius)
    # Plot it also, when both sides are exceeded
    if all(center + radius > width):
        n_center = center.copy() - width
        plot_circle(n_center, radius=radius)


def plot_circle(center, radius):
    circle = mpatches.Circle(center, radius=radius, fc="None", ec="black", linewidth=2, ls="dashed")
    plt.gca().add_artist(circle)



if __name__ == "__main__":
    main()
