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
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import copy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from figure_generator.figure3 import panel_avg_activity
from figure_generator.figure1 import xyticks
from plot.constants import *
from lib.neuralhdf5 import NeuralHdf5, default_filename
import lib.universal as UNI
from plot.lib import remove_spines_and_ticks, add_topright_spines
import lib.pickler as PIC
from params import RandomConfig
config = RandomConfig() # Use a specific one here!
#===============================================================================
# CONSTANTS
#===============================================================================

filename = "supp_intersect"


figsize = (17.6*cm, 6*cm)

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=(0.2, 1, 1))
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.16,
        top=0.84,
        wspace=0.0,
    )
    
    ##### TOP LEFT ##############################
    
    name, p = "select-right", -.1
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(f"Activity Difference\n(Select) B2: {p:+.0%}")
    add_topright_spines(ax)
    for spine in ax.spines.values():
        spine.set_edgecolor("tab:olive")
    ax.set(xlabel="X", xticks=(30, 60, 90), xlim=(28, 93))
    ax.set(ylabel="Y", yticks=(10, 30, 50), ylim=(0, 65))
    ax, cbar = panel_avg_activity(ax, config, name=name, p=p)
    cbar.set_label(r"$\Delta$ avg. activity", rotation=270, labelpad=6)
    
    
    name, p = "gate-left", .1
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title(f"Activity Difference\n(Gate) B1: {p:+.0%}")
    add_topright_spines(ax)
    for spine in ax.spines.values():
        spine.set_edgecolor("tab:cyan")
    shifted_ticks = ((10, 30, 49, 50, 70, 90), (60, 80, 99, " ", 20, 40))
    ax.set_yticks(*shifted_ticks)
    ax.set(xlabel="X", xticks=(10, 40, 70), xlim=(8, 73))
    ax.set(ylabel="Y", ylim=(20, 85))
    _, cbar = panel_avg_activity(ax, config, name=name, p=p, roll=(50, 0))
    cbar.set_label(r"$\Delta$ avg. activity", rotation=270, labelpad=6)

    # ax = fig.add_axes((-.01, .4, .3, .3))
    ax = fig.add_subplot(gs[0, 0])
    add_topright_spines(ax)
    from figure_generator.figure1 import panel_avg_activity as paa
    ax.set(xticks=(), yticks=())
    ax.tick_params(labelleft=False, labelbottom=False)
    im = paa(ax, config)
    
    from matplotlib.patches import Rectangle
    rect_kwargs = {"fc": "none", "lw": 2}
    rect = Rectangle((28, 0), 65, 65, ec="tab:olive", **rect_kwargs, zorder=5)
    ax.add_patch(rect)
    rect = Rectangle((8, 35), 65, -65, ec="tab:cyan", **rect_kwargs)
    ax.add_patch(rect)
    rect = Rectangle((8, 70), 65, 65, ec="tab:cyan", **rect_kwargs)
    ax.add_patch(rect)
    
    
    PIC.save_figure(filename, fig, transparent=True)
#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
