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


from plot.lib.frame import create_image
from plot.lib.basic import add_colorbar, plot_patch_from_tag
from plot.constants import COLOR_MAP_ACTIVITY, NORM_ACTIVITY, COLOR_MAP_DIFFERENCE


#===============================================================================
# METHODS
#===============================================================================

def baseline_average(config:object):
    tags = config.baseline_tags

    # Gather all rates
    rates = []
    for tag in tags:
        logger.info(f"Load {tag}...")
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)
        rates.append(avgRate)
    rates = np.asarray(rates)
    print(rates.shape)

    # Average if more than one run
    if rates.ndim > 1:
        rates = rates.mean(axis=0)


    figname = "baseline_averaged_across_seeds"

    fig, ax = plt.subplots(num=figname,
                           figsize=(5.5, 3.5),
                           # tight_layout=True,
                           )
    norm = (0, 0.5)
    cmap = COLOR_MAP_ACTIVITY

    create_image(rates, norm, cmap, axis=ax)

    cbar = add_colorbar(ax, norm, cmap)
    cbar.set_label("Avg. activity [a.u.]", rotation=270, labelpad=15)
    cbar.set_ticks([0.0, 0.2, 0.4])

    # plt.title("Avg. activity across seeds")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([10, 40, 70])
    ax.set_yticks([10, 40, 70])
    # plt.tight_layout()

    from plot.lib import plot_patch
    for name, center in config.center_range.items():
        plot_patch(center, config.radius[0], width=config.rows, axis=ax)
        ax.text(*center, name, verticalalignment="center", horizontalalignment="center", zorder=12)

    PIC.save_figure(figname, fig, sub_directory=config.sub_dir, transparent=True)



def avg_activity(postfix:list, config:object) -> None:
    postfix = UNI.make_iterable(postfix)

    for tag in postfix:
        logger.info(f"Load {tag}...")
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)


        fig, ax = plt.subplots(num=tag)
        norm = NORM_ACTIVITY
        cmap = COLOR_MAP_ACTIVITY
        im = create_image(avgRate, norm=norm, cmap=cmap)
        # plt.title("Avg. activity")
        cbar = add_colorbar(ax, norm, cmap)
        cbar.set_label("Avg. activity [a.u.]", rotation=270, labelpad=15)

        try:
            plot_patch_from_tag(tag, config)
        except KeyError:
            logger.info(f"Could not find patch for tag: {tag}")


        PIC.save_figure(filename=tag, figure=fig, sub_directory=config.sub_dir)
        plt.title((avgRate).mean())
