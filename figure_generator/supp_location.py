#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: 
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import copy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from plot.constants import *
from lib.neuralhdf5 import NeuralHdf5, default_filename
import lib.universal as UNI
import lib.pickler as PIC
from params import RandomConfig
config = RandomConfig() # Use a specific one here!
#===============================================================================
# CONSTANTS
#===============================================================================

filename = "supp_location"
from figure_generator.figure2 import panel_random_patch_locations, add_density_contourf
from figure_generator.figure2 import min_degree, max_degree, ylim_duration, xlim_seq, title_style, xticks_seq, yticks_duration

figsize = (17.6*cm, 6*cm)

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=1, ncols=5, width_ratios=(1, 1, .35, 1, 1))
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.16,
        top=0.9,
        wspace=0.15,
    )
    
    ##### TOP LEFT: RANDOM PATCH LOCATIONS ##############################
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_pos.set(title="Patch: +10%", xlabel="Sequence count", ylabel="Avg. duration [ms]",
               xticks=xticks_seq, yticks=yticks_duration, xlim=xlim_seq, ylim=ylim_duration)
    
    ax_neg = fig.add_subplot(gs[0, 1])
    ax_neg.set(title="Patch: -10%", xlabel="Sequence count", #ylabel="Avg. duration [ms]",
               xticks=xticks_seq, yticks=yticks_duration, xlim=xlim_seq, ylim=ylim_duration)
    ax_neg.tick_params(labelleft=False)
    
    panel_random_patch_locations(ax_pos, config, p=0.1)
    add_density_contourf(ax_pos, config, p=0.1)
    panel_random_patch_locations(ax_neg, config, p=-0.1)
    add_density_contourf(ax_neg, config, p=-0.1)
    ax_neg.legend()
    
    ##### TOP RIGHT: RANDOM NEURONS ######################################
    ax_pos = fig.add_subplot(gs[0, -2])
    ax_pos.set(title="Neurons: +10%", xlabel="Sequence count", ylabel="Avg. duration [ms]",
               xticks=xticks_seq, yticks=yticks_duration, xlim=xlim_seq, ylim=ylim_duration)
    
    ax_neg = fig.add_subplot(gs[0, -1])
    ax_neg.set(title="Neurons: -10%", xlabel="Sequence count", #ylabel="Avg. duration [ms]",
               xticks=xticks_seq, yticks=yticks_duration, xlim=xlim_seq, ylim=ylim_duration)
    ax_neg.tick_params(labelleft=False)
    
    tmp_config = copy.copy(config)
    tmp_radius = config.radius[0]
    tmp_config.radius = 80, 
    panel_random_patch_locations(ax_pos, tmp_config, p=0.1)
    add_density_contourf(ax_pos, tmp_config, p=0.1)
    panel_random_patch_locations(ax_neg, tmp_config, p=-0.1)
    add_density_contourf(ax_neg, tmp_config, p=-0.1)
    assert config.radius[0] == tmp_radius
    ax_neg.legend()
    
    PIC.save_figure(filename, fig, transparent=True)
#===============================================================================
# METHODS
#===============================================================================

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
