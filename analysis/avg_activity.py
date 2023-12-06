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
from cflogger import logger

import lib.pickler as PIC
from plot.avg_activity import plot_avg_activity
from plot.animation import animate
from plot.activity_difference import plot_activity_differences

from params import config

# Refactor: Put this to Analysis parameter
AVERAGE_BASELINE_RATES = True
AVERAGE_RATES = True


#===============================================================================
# MAIN AND TESTING AREA
#===============================================================================
def main():
    if AVERAGE_BASELINE_RATES:
        logger.info(f"Average rates: {config.baseline_tags}")
        _average_rate(*config.baseline_tags, sub_directory=config.sub_dir, config=config)
    if AVERAGE_RATES:
        tags = config.get_all_tags()
        logger.info(f"Average rates: {tags}")
        _average_rate(*tags, sub_directory=config.sub_dir, config=config)


    _request_plot = input("Do you want to plot the averages? (y: all; p:patches only; bs:baselines only)").lower()
    if _request_plot == "y":
        plot_avg_activity(config, plot_baseline_average=True, baseline_seeds=True, patches_seeds=True)
    elif _request_plot == "p":
        plot_avg_activity(config, plot_baseline_average=False, baseline_seeds=False, patches_seeds=True)
    elif _request_plot == "bs":
        plot_avg_activity(config, plot_baseline_average=True, baseline_seeds=True, patches_seeds=False)

    _request_plot_differences = input("Do you want to plot the average differences? (y: all; p:patches only; bs:baselines only)").lower()
    if _request_plot_differences == "y":
        plot_activity_differences(config, patch_vs_baseline=True, baseline_across_seeds=True)
    elif _request_plot_differences == "p":
        plot_activity_differences(config, patch_vs_baseline=True, baseline_across_seeds=False)
    elif _request_plot_differences == "bs":
        plot_activity_differences(config, patch_vs_baseline=False, baseline_across_seeds=True)

    _request_animation = input("Do you want to animate the rates? (y: all; p:patches only; bs:baselines only, d:baseline differences)").lower()
    if _request_animation == "y":
        animate(config, animate_baseline=True, animate_patch=True)
    elif _request_animation == "p":
        animate(config, animate_baseline=False, animate_patch=True)
    elif _request_animation == "bs":
        animate(config, animate_baseline=True, animate_patch=False)
    elif _request_animation == "d":
        animate(config, animate_baseline=False, animate_patch=False, animate_baseline_differences=True)



#===============================================================================
# METHODS
#===============================================================================


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





if __name__ == '__main__':
    main()
