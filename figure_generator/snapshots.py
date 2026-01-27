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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from params import config

from lib import pickler as PIC
from lib import universal as UNI
from lib import dopamine as DOP

from lib.neuralhdf5 import NeuralHdf5, default_filename

#===============================================================================
# CONSTANTS
#===============================================================================


#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    no_snapshots = 5
    tag = config.baseline_tags[3]
    # tag = config.get_all_tags("select-left", seeds=1, weight_change=.1)[0]
    tags = config.baseline_tags[3:4]
    for tag in tags:
        print(tag)
        with NeuralHdf5(default_filename, "a", config=config) as file:
            # spikes, labels = file.get_spikes_with_labels(tag, is_baseline=False)
            spikes, labels = file.get_spikes_with_labels(tag, is_baseline=True)
            rate = PIC.load_rate(tag, skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)
            
        # 1 index is time
        fig, axes = plt.subplots(nrows=2, ncols=no_snapshots, sharex=True, sharey=True, figsize=(6, 4))
        axes[0, 0].set_ylabel("y")
        axes[0, 0].set_yticks([10, 40, 70])
        axes[1, 0].set_ylabel("y")
        axes[1, 0].set_yticks([10, 40, 70])
        for i in range(no_snapshots):
            axes[1, i].set_xlabel("x")
            axes[1, i].set_xticks([10, 40, 70])
        
        t_start = 190
        t_step  = 20
        t_stop  = t_start + no_snapshots*t_step
        
        
        from plot.lib.frame import create_image
        from plot.lib import add_colorbar
        im_kwargs = {"cmap": "hot_r", "norm": (0, 1)}
        for i in range(no_snapshots):
            snap = rate[:, t_start+i*t_step:t_start+(i+1)*t_step].mean(axis=1)
            im = create_image(snap, **im_kwargs, axis=axes[0, i])
            # axes[0, i].imshow(snap.reshape((config.rows, config.rows)), **im_kwargs)
    
        cb_ax = fig.add_axes([.88,.52,.01,.36])
        cbar = fig.colorbar(im, orientation='vertical', cax=cb_ax)
        cbar.set_label("rate [au]", rotation=270)
        
        
    
        import matplotlib as mpl
        cmap = mpl.colormaps.get_cmap('jet')  # viridis is the default colormap for imshow
        cmap.set_bad(color='white')
        for i in range(no_snapshots):
            spikes_tmp = spikes[np.logical_and(spikes[:, 0] >= t_start+i*t_step, spikes[:, 0] < t_start+(i+1)*t_step)]
            
            spikes_at_location = np.zeros((config.rows, config.rows))
            for t, x, y in spikes_tmp:
                if spikes_at_location[x, y] < t:
                    spikes_at_location[x, y] = t
            spikes_at_location[spikes_at_location == 0] = np.nan
            
            im = create_image(spikes_at_location.T, norm=(t_start, t_stop), cmap=cmap, axis=axes[1, i])
            
        
        cb_ax = fig.add_axes([.88,.1,.01,.36])
        cbar = fig.colorbar(im, orientation='vertical', cax=cb_ax)
        cbar.set_label("time [ms]", rotation=270)
        # add_colorbar(axes[1, no_snapshots-1],norm=(t_start, t_stop), cmap=cmap)
#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
