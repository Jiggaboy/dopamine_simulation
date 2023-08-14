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

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable

import lib.pickler as PIC
import lib.universal as UNI
from plot.avg_activity import plot_avg_activity
from plot.animation import animate

from params import BaseConfig, BrianConfig, PerlinConfig, NullConfig, ScaleupConfig, StarterConfig, LowDriveConfig

# Refactor: Put this to Analysis parameter
AVERAGE_BASELINE_RATES = True
AVERAGE_RATES = True

Config = BrianConfig()
tag_subset = None


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    if AVERAGE_BASELINE_RATES:
        logger.info(f"Average rates: {Config.baseline_tags}")
        _average_rate(*Config.baseline_tags, sub_directory=Config.sub_dir, config=Config)
    if AVERAGE_RATES:
        tags = Config.get_all_tags(tag_subset)
        logger.info(f"Average rates: {tags}")
        _average_rate(*tags, sub_directory=Config.sub_dir, config=Config)


    _request_plot = input("Do you want to plot the averages? (y: all; p:patches only; bs:baselines only)").lower()
    if _request_plot == "y":
        plot_avg_activity(Config, plot_baseline_average=True, baseline_seeds=True, patches_seeds=True)
    elif _request_plot == "p":
        plot_avg_activity(Config, plot_baseline_average=False, baseline_seeds=False, patches_seeds=True)
    elif _request_plot == "bs":
        plot_avg_activity(Config, plot_baseline_average=True, baseline_seeds=True, patches_seeds=False)

    _request_animation = input("Do you want to animate the rates? (y: all; p:patches only; bs:baselines only, d:baseline differences)").lower()
    if _request_animation == "y":
        animate(Config, animate_baseline=True, animate_patch=True)
    elif _request_animation == "p":
        animate(Config, animate_baseline=False, animate_patch=True)
    elif _request_animation == "bs":
        animate(Config, animate_baseline=False, animate_patch=True)
    elif _request_animation == "d":
        animate(Config, animate_baseline=False, animate_patch=False, animate_baseline_differences=True)



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
