#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 20:54:57 2022

@author: hauke
"""

import cflogger
logger = cflogger.getLogger()

from collections.abc import Iterable
import matplotlib.pyplot as plt
import numpy as np

import util.pickler as PIC
import universal as UNI

from plot.lib import plot_activity
from figure_generator.connectivity_distribution import set_layout

## Specifiy the Config here
from params import PerlinConfig

def main():
    cf = PerlinConfig()
    avg_activity(cf.baseline_tag, cf)
    all_tags = cf.get_all_tags()
    print(all_tags)
    avg_activity(all_tags, cf)


def avg_activity(postfix, config)->None:
    postfix = UNI.make_iterable(postfix)

    for tag in postfix:
        logger.info(f"Load {tag}...")
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)

        plot_activity(avgRate, norm=(0, .5), figname=tag, figsize=(7, 6))
        plt.title("Avg. activity")
        #############
        # Make Details of the figure here!
        set_layout(config.rows, margin=0)
        plt.savefig(UNI.get_fig_filename(tag + "_avg", format_="svg"), format="svg")
        plt.title((avgRate).mean())
        
        
def patchy_activity(activity:np.ndarray, patch:np.ndarray)->None:
    """
    activity, patch:
        2D array
    """
    plot_activity(activity[~patch], tag="patched_activity")
    set_layout(70, margin=0)
    
    

if __name__ == "__main__":
    main()
    plt.show()
