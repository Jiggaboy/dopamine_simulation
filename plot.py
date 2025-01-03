#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Aggregates all plots.

Description:
    - Definition of control structeres
    - Plotting methods are in the module 'plot'


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

import numpy as np
import matplotlib.pyplot as plt

from params import config

import lib.universal as UNI

from plot.figconfig import ActivityDifferenceConfig, AnimationConfig
from plot.activity_difference import Plot_ActivityDifference
from plot.animation import Animator
import plot.avg_activity as avg_activity

### Average Activity
baseline_average_across_seeds = UNI.yes_no("Average Activity: Baseline (across seeds)?", False)
baseline_average_per_seed = UNI.yes_no("Average Activity: Baseline (split by seed)?", False)
patch_average_per_seed = UNI.yes_no("Average Activity: Patches (split by seed)?", False)

### Activity Differences
config_activity_difference = ActivityDifferenceConfig
patch_vs_baseline_activity = UNI.yes_no("Activity difference: Plot Patch vs baseline?", True)
baseline_across_seeds_difference = UNI.yes_no("Activity difference: Plot Baselines across seeds?", False)

### Activity Animation
config_animation = AnimationConfig
animate_baseline = UNI.yes_no("Animation: Animate Baseline (Seed: 0)?", False)
animate_patch = UNI.yes_no("Animation: Animate Patches (Seed: 0)?", False)
animate_baseline_differences = UNI.yes_no("Animation: Baseline_differences?", False)


#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    ### Average Activity
    # Average activity of all baseline simulations (averaged)
    if baseline_average_across_seeds:
        avg_activity.baseline_average(config)
    # Average activity of a baseline simulation (individual runs)
    if baseline_average_per_seed:
        all_tags = config.baseline_tags
        avg_activity.avg_activity(all_tags, config)
    # Average activity of the patches (Similar figures to the baseline ones)
    if patch_average_per_seed:
        all_tags = config.get_all_tags()
        avg_activity.avg_activity(all_tags, config)


    ### Activity Difference
    activity_difference = Plot_ActivityDifference(config, config_activity_difference)
    if patch_vs_baseline_activity:
        # Plots the individual runs against the baseline, but also the average.
        tags = config.get_all_tags(seeds="all")
        activity_difference.activity_difference(tags)
    if baseline_across_seeds_difference:
        activity_difference.baseline_difference_across_seeds()


    ### Activity Animation
    animator = Animator(config, config_animation)
    if animate_baseline:
        animator.animate(config.baseline_tags[:1])
    if animate_patch:
        animator.animate(config.get_all_tags(seeds=0))
    if animate_baseline_differences:
        animator.baseline_difference_animations()




#===============================================================================
# METHODS
#===============================================================================






if __name__ == '__main__':
    main()
    plt.show()
