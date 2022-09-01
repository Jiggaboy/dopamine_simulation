#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:20:21 2022

@author: hauke
"""

import matplotlib.pyplot as plt
import numpy as np

from plot.lib import SequenceCounter

import util.pickler as PIC
import universal as UNI


from figure_generator.connectivity_distribution import set_layout, SPINE_WIDTH

from plot.lib import image_slider_1d

MS = 8
DISTANCE_BETWEEN_SCATTERS = 0.1

KTH_GREEN = 176, 201, 43
KTH_PINK = 216, 84, 151
KTH_GREY = 101, 101, 108
colors = KTH_GREEN, KTH_PINK, KTH_GREY
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


BS_SEQ_FIGSIZE = (12, 6)
    
def main():
    from params import PerlinConfig

    cf = PerlinConfig()

    all_tags = cf.get_all_tags()


    
    plot_baseline_sequences(config=cf)
    
    slider = []
    for tag in all_tags:
        fig, (ax_seq, ax_db_seq) = plt.subplots(ncols=2, num=tag, figsize=(8, 4))
        fig.suptitle("# sequences with 'old' method and DBscan")
        for ax in (ax_seq, ax_db_seq):
            ax.set_xlim(-.1, 1.6)
            ax.set_ylim(0, 80)
        
        plot_sequences(cf, tag, load_method=PIC.load_sequence, axis=ax_seq)
        plot_sequences(cf, tag, load_method=PIC.load_db_sequence, axis=ax_db_seq)
        
        #from functools import partial
        #method = partial(update_sequence, axis=ax_seq)
        #sequences = PIC.load_sequence(tag, sub_directory=cf.sub_dir)
        #s = image_slider_1d(sequences, fig, axis=ax_seq, method=method, label="Seed")
        #slider.append(s)
        PIC.save_figure(f"seq_compare_{tag}", fig, cf.sub_dir)
    plt.show()

    
def plot_sequences(config:object, tag:str, load_method, axis, **plot_kwargs):
    sequence = load_method(tag, sub_directory=config.sub_dir)
    handles = []
    for idx, (center, c) in enumerate(zip(sequence.center, colors)):
        c = np.asarray(c) / 255
        handle = scatter_baseline_patch(idx * DISTANCE_BETWEEN_SCATTERS, sequence, idx, c=c, axis=axis, **plot_kwargs)
        handles.append(handle)
    axis.set_ylabel("# Sequences")
    axis.set_xticks([.1, 1.1], labels=["w/o patch", "w/ patch"])
    axis.legend(handles=handles, labels=sequence.center, loc="lower center")
            

#def update_sequence(sequence, axis, idx):
#    for idx, (center, c) in enumerate(zip(sequence.center, colors)):
 #       c = np.asarray(c) / 255
  #      scatter_baseline_patch(idx * DISTANCE_BETWEEN_SCATTERS, sequence, idx, c=c, axis=axis, **plot_kwargs)



def plot_baseline_sequences(config:object)->None:
    """
    Loads the # of sequences detected by DBScan.
    """
    fig, axes = plt.subplots(ncols=len(config.simulation_seeds), num="baseline_sequences", figsize=BS_SEQ_FIGSIZE)
    fig.suptitle("# Sequences across baselines with different center (DBScan)")
    for seed in config.simulation_seeds:
        tag = config.baseline_tag(seed)
        sequence = PIC.load_db_sequence(tag, sub_directory=config.sub_dir)
        handles = []
        for idx, (center, color) in enumerate(zip(sequence.center, color_cycle)):
            handle = scatter_baseline(idx, sequence, idx, c=color, axis=axes[seed])
            handles.append(handle)

        axes[seed].set_ylabel("# Sequences")
    axes[seed].legend(handles=handles, labels=sequence.center)
    PIC.save_figure("seq_across_baselines", fig, config.sub_dir)
        

def scatter(x, data, axis:object=None, **kwargs):
    ax = axis if axis is not None else plt
    plot_to_scatter = {"ls": "None", "marker": "o"}
    try:
        line, = ax.plot(np.full(shape=len(data), fill_value=x), data, **plot_to_scatter, **kwargs)
    except TypeError:
        line, = ax.plot(x, data, **plot_to_scatter, **kwargs)
    return line


def scatter_baseline_patch(x, sequence, center_idx:int, distance:float=1., **kwargs):
    scatter(x, sequence.baseline[center_idx], markerfacecolor="white", markersize=MS / 2, **kwargs)
    scatter(x+distance, sequence.patch[center_idx], markerfacecolor="white", markersize=MS / 2, **kwargs)

    scatter(x, sequence.baseline_avg[center_idx], markersize=MS, **kwargs)
    return scatter(x+distance, sequence.patch_avg[center_idx], markersize=MS, **kwargs)

def scatter_baseline(x, sequence, center_idx:int, distance:float=1., **kwargs):
    scatter(x, sequence.baseline[center_idx], markerfacecolor="white", markersize=MS / 2, **kwargs)
    return scatter(x, sequence.baseline_avg[center_idx], markersize=MS, **kwargs)


def bold_spines(ax, width:float=SPINE_WIDTH):
    # ax.set_xticks([])
    tick_params = {"width": SPINE_WIDTH, "length": SPINE_WIDTH * 3, "labelleft": True, "labelbottom": True}
    ax.tick_params(**tick_params)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('bottom', 'left'):
            ax.spines[s].set_linewidth(SPINE_WIDTH)


if __name__ == "__main__":
    main()
