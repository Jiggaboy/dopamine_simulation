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
from plot.sequences import plot_sequence_landscape, plot_count_and_duration, plot_seq_diff, plot_seq_duration_over_indegree

### Average Activity
baseline_average_across_seeds = UNI.yes_no("Average Activity: Baseline (across seeds)?", False)
baseline_average_per_seed = UNI.yes_no("Average Activity: Baseline (split by seed)?", False)
patch_average_per_seed = UNI.yes_no("Average Activity: Patches (split by seed)?", False)

### Activity Differences
config_activity_difference = ActivityDifferenceConfig
patch_vs_baseline_activity = UNI.yes_no("Activity difference: Plot Patch vs baseline?", False)
baseline_across_seeds_difference = UNI.yes_no("Activity difference: Plot Baselines across seeds?", False)

### Activity Animation
config_animation = AnimationConfig
animate_baseline = UNI.yes_no("Animation: Animate Baseline (Seed: 0)?", False)
animate_patch = UNI.yes_no("Animation: Animate Patches (Seed: 0)?", False)
animate_baseline_differences = UNI.yes_no("Animation: Baseline_differences?", False)

### Sequences
plot_sequence_count_on_location = UNI.yes_no("Sequences: Plot Sequence count on location?", False)
plot_sequence_count_and_duration = UNI.yes_no("Sequences: Plot sequence count and duration?", False)
plot_sequence_count_difference = UNI.yes_no("Sequences: Plot difference across sequence counts?", False)
plot_sequences_over_indegree = UNI.yes_no("Sequences: Plot sequence duration over indegree?", True)


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


    # Plots the number of detected sequences on the grid.
    if plot_sequence_count_on_location:
        tags = config.get_all_tags()
        # Plots only baseline if no tags are detected
        if tags == []:
            for tag in config.baseline_tags:
                plot_sequence_landscape(tag, config)
        else:
            for tag in tags[:]:
                tag_tmp = config.get_baseline_tag_from_tag(tag)
                plot_sequence_landscape(tag_tmp, config)
                plot_sequence_landscape(tag, config)

    if plot_sequence_count_and_duration:
        plot_count_and_duration(config)

    if plot_sequence_count_difference:
        plot_seq_diff(config)

    if plot_sequences_over_indegree:
        plot_seq_duration_over_indegree(config, feature="duration")
        plot_seq_duration_over_indegree(config, feature="sequence")

#===============================================================================
# METHODS
#===============================================================================






if __name__ == '__main__':
    main()
    plt.show()
