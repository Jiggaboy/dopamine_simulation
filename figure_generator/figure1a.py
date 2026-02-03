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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from params import config
import lib.pickler as PIC
from plot.lib.frame import create_image
from plot.lib.basic import add_colorbar, plot_patch_from_tag, plot_patch, add_topright_spines
from plot.constants import COLOR_MAP_ACTIVITY, NORM_ACTIVITY, COLOR_MAP_DIFFERENCE, cm, CMAP_DEGREE, title_style
from lib.neuralhdf5 import NeuralHdf5, default_filename
from figure_generator.figure1 import xyticks

from plot.lib import add_colorbar

#===============================================================================
# CONSTANTS
#===============================================================================

figsize = (17.6*cm, 10*cm)

im_kwargs = {"cmap": COLOR_MAP_ACTIVITY, "norm": (0, .5)}

spikes_kwargs = {"cmap": CMAP_DEGREE}



filename="snapshots"
    
xc_offset = 0.02
cbar_kwargs = {"rotation":-90, "labelpad": 12}
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    no_snapshots = 3
    tag = config.baseline_tags[0]
    # tag = config.get_all_tags("start-1", seeds=0)[0]
    print(tag)
    
    t_start = 700
    # t_start = 2300
    t_step  = 50
    t_stop  = t_start + no_snapshots*t_step
    ticks_time = np.arange(t_start, t_stop+1, t_step, dtype=int)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=no_snapshots, )#width_ratios=[*np.ones(no_snapshots-1), 1.06])
    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.92,
        wspace=0.1,
        hspace=0.1, 
    )

    
    # Collect Spikes and rate
    with NeuralHdf5(default_filename, "a", config=config) as file:
        # spikes, labels = file.get_spikes_with_labels(tag, is_baseline=False)
        spikes, labels = file.get_spikes_with_labels(tag, is_baseline=True)
    rate = PIC.load_rate(tag, skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)
        
    row = 0
    for i in range(no_snapshots):
        ax = fig.add_subplot(gs[row, i])
        add_topright_spines(ax)
        ax.set_title(fr"{t_start+i*t_step}ms$\leqslant$t$<${t_start+(i+1)*t_step}ms")
        ax.set_xticks(xyticks)
        ax.tick_params(labelbottom=False)
        ax.set_yticks(xyticks)
        if i == 0:
            bbox = ax.get_position()
            xc = bbox.x0 / 2 - xc_offset
            yc = (bbox.y0 + bbox.y1) / 2
            fig.text(xc, yc, "Average Rate", title_style, rotation=90)
            ax.set_ylabel("Y")
        else:
            ax.tick_params(labelleft=False)
        
        snap = rate[:, t_start+i*t_step:t_start+(i+1)*t_step].mean(axis=1)
        im = create_image(snap, **im_kwargs, axis=ax)

    cbar = add_colorbar(ax, **im_kwargs)
    cbar.set_ticks((0, .2, .4))
    cbar.set_label("Rate", **cbar_kwargs)
    
    
    # Plot the spikes
    spikes_kwargs["norm"] = (t_start, t_stop)
    row = 1
    for i in range(no_snapshots):
        ax = fig.add_subplot(gs[row, i])
        add_topright_spines(ax)
        # ax.set_title("Detected Spikes")
        ax.set_xlabel("X")
        ax.set_xticks(xyticks)
        ax.set_yticks(xyticks)
        if i == 0:
            bbox = ax.get_position()
            xc = bbox.x0 / 2 - xc_offset
            yc = (bbox.y0 + bbox.y1) / 2
            fig.text(xc, yc, "Detected Spikes", title_style, rotation=90)
            ax.set_ylabel("Y")
        else:
            ax.tick_params(labelleft=False)
            
        spikes_tmp = spikes[np.logical_and(spikes[:, 0] >= t_start+i*t_step, spikes[:, 0] < t_start+(i+1)*t_step)]
        
        spikes_at_location = np.zeros((config.rows, config.rows))
        for t, x, y in spikes_tmp:
            if spikes_at_location[x, y] < t:
                spikes_at_location[x, y] = t
        spikes_at_location[spikes_at_location == 0] = np.nan
        
        im = create_image(spikes_at_location.T, **spikes_kwargs, axis=ax)
        
    
    cbar = add_colorbar(ax, **spikes_kwargs)
    cbar.set_ticks(ticks_time)
    cbar.set_label("Time [ms]", **cbar_kwargs)
    
    
    PIC.save_figure(filename, fig, transparent=True)
    
#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
