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

from plot.figconfig import ActivityDifferenceConfig
from plot.activity_difference import Plot_ActivityDifference



### Activity differences
config_activity_difference = ActivityDifferenceConfig
patch_vs_baseline_activity = UNI.yes_no("Activity difference: Plot Patch vs baseline?")

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():

    ### Activity differences
    activity_difference = Plot_ActivityDifference(config, config_activity_difference)
    if patch_vs_baseline_activity:
        tags = config.get_all_tags(seeds="all")
        activity_difference.activity_difference(tags)
    # if baseline_across_seeds:
    #     activity_difference.baseline_difference_across_seeds()





#===============================================================================
# METHODS
#===============================================================================






if __name__ == '__main__':
    main()
