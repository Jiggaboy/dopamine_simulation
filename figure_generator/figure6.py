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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from params import config
import lib.pickler as PIC
from plot.lib.frame import create_image
from plot.lib.basic import add_colorbar, plot_patch_from_tag, plot_patch
from plot.constants import COLOR_MAP_ACTIVITY, NORM_ACTIVITY, COLOR_MAP_DIFFERENCE, cm
from lib.neuralhdf5 import NeuralHdf5, default_filename

from plot.lib import add_colorbar_from_im
import lib.universal as UNI

from task import TaskConfig, CustomConnectivityMatrix
config = TaskConfig()
#===============================================================================
# CONSTANTS
#===============================================================================
rcParams["font.size"] = 8
rcParams["figure.figsize"] = (17.6*cm, 12*cm)
rcParams["legend.fontsize"] = 7
rcParams["legend.markerscale"] = 0.6
rcParams["legend.handlelength"] = 1.25
rcParams["legend.columnspacing"] = 1
rcParams["legend.handletextpad"] = 1
rcParams["legend.labelspacing"] = .1
rcParams["legend.borderpad"] = .25
rcParams["legend.handletextpad"] = .5
rcParams["legend.framealpha"] = 1
rcParams["axes.labelpad"] = 2

im_kwargs = {"cmap": "hot_r", "norm": (0, .6)}

spikes_kwargs = {"cmap": "jet"}

title_style = {
    "fontsize": plt.rcParams["axes.titlesize"],
    "fontweight": plt.rcParams["axes.titleweight"],
    "fontfamily": plt.rcParams["font.family"],
    "ha": "center",
    "va": "center"
}

filename = "task"
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure()
    
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1.1])

    gs_top = gs[0].subgridspec(nrows=1, ncols=4, width_ratios=[3.4, 0, 1, 1])
    gs_bottom = gs[1].subgridspec(nrows=1, ncols=4, width_ratios=[2, 1., .75, 1.94])
    
    # gs = fig.add_gridspec(nrows=2, ncols=5, 
    #                       width_ratios=(2, 1.4, .4, 1, 1), height_ratios=(1, 1))
    fig.subplots_adjust(
        left=0.08,
        right=0.96,
        bottom=0.14,
        top=0.91,
        wspace=0.0125,
        hspace=0.45, 
    )
    idx_indegree = 1
    idx_sequence_count_bs = 2
    idx_sequence_count_patch = -1
    amount = config.AMOUNT_NEURONS[0]
    radius = config.radius[0]
    pseudopercent = 0 # actual percentages are defined within the task
    seed = 0
    taskname = "task-A"
    assert taskname in config.task.keys()
    tag_patch = UNI.get_tag_ident(taskname, radius, amount, pseudopercent, seed)
    tag_bs = config.baseline_tag(seed)
    maxpercent = 0.2
    
    # Network Schematic placeholder
    # ax = fig.add_subplot(gs[:, 0])

    
    # Task Schematic placeholder
    # ax = fig.add_subplot(gs[0, 1])

    # INDEGREE
    # ax = fig.add_subplot(gs[1, idx_indegree])
    ax = fig.add_subplot(gs_bottom[idx_indegree])
    ax.set_title("In-Degree")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xticks([30, 50, 70])
    ax.set_xlim([25, 75])
    panel_indegree(ax, config)

    # SEQUENCE COUNT
    # ax = fig.add_subplot(gs[0, idx_sequence_count_bs])
    ax_baseline = fig.add_subplot(gs_top[idx_sequence_count_bs])
    ax_baseline.set_title("Baseline")
    # ax_baseline.set_title("Sequence Count\nBaseline")
    ax_baseline.set_xlabel("X")
    ax_baseline.set_ylabel("Y")
    ax_baseline.set_xticks([30, 50, 70])
    ax_baseline.set_xlim([25, 75])
    # ax_baseline.tick_params(labelleft=False)
    ax_baseline.set_yticks([10, 50, 90])
    ax_baseline.set_ylim([0, 90])
    
    panel_sequence_count(ax_baseline, config, tag_bs, is_baseline=True)
    
    # ax = fig.add_subplot(gs[0, idx_sequence_count_patch])
    ax_patch = fig.add_subplot(gs_top[idx_sequence_count_patch])
    # ax_patch.set_title("Sequence Count\nPatches")
    ax_patch.set_title("Patches")
    ax_patch.set_xlabel("X")
    ax_patch.set_xticks([30, 50, 70])
    ax_patch.set_xlim([25, 75])
    ax_patch.tick_params(labelleft=False)
    ax_patch.set_yticks([10, 50, 90])
    ax_patch.set_ylim([0, 90])
    
    im = panel_sequence_count(ax_patch, config, tag_patch)
    cbar = add_colorbar_from_im(ax_patch, im)
    # cbar.set_ticks(())
    cbar.set_label("Sequence count", rotation=270, labelpad=10)
        
    for name, percent in config.task[taskname]:
        c = config.center_range[name]
        ec = "red" if percent < 0 else "green"
        alpha = abs(percent / maxpercent) 
        plot_patch(c, radius, width=config.rows, axis=ax_patch, ec=ec, alpha=alpha)
    
    offset = 5
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tag_patch)
    for s, ds in enumerate(detection_spots):
        plot_patch(ds, radius=2., width=config.rows, axis=ax_patch, lw=1)
        plot_patch(ds, radius=2., width=config.rows, axis=ax_baseline, lw=1)
        if s < 3:
            text = f"C{s}"
        elif s == 3:
            text = "L"
        elif s == 4:
            text = "R"
        else:
            raise ValueError
        ax_patch.text(ds[0]+offset, ds[1], text, verticalalignment="center")
        ax_baseline.text(ds[0]+offset, ds[1], text, verticalalignment="center")

    # SEQUENCE CORRELATION
    # ax = fig.add_subplot(gs[1, 2:])
    ax = fig.add_subplot(gs_bottom[-1])
    ax.set_title("Sequence Correlations")
    panel_sequence_correlations(ax, config, tag_patch, taskname)
    ax.set_ylabel("Sequence count", labelpad=0)
    ax.set_yticks((0, 5, 10, 15))
    ax.set_ylim((0, 20))
    ax.legend(ncols=2, loc="upper center")

    PIC.save_figure(filename, fig)
#===============================================================================
# METHODS
#===============================================================================

def panel_indegree(ax:object, config:object):
    conn = CustomConnectivityMatrix(config)
    indegree, _ = conn.degree(conn._EE)
    indegree = indegree * config.synapse.weight

    cmap = plt.cm.jet
    im = ax.imshow(indegree,
                    origin="lower",
                    cmap=cmap,
    )
    cbar = add_colorbar_from_im(ax, im)
    cbar.set_ticks([800, 1000, 1200])
    cbar.set_label("In-degree", rotation=270, labelpad=8)


def panel_sequence_count(ax:object, config:object, tag:str, **kwargs):
    from plot.sequences import _get_sequence_landscape    

    spikes, labels = PIC.load_spike_train(tag, config)
    sequence_landscape = _get_sequence_landscape(spikes, labels, config.rows)
    from plot.sequences import truncate_colormap
    cmap = truncate_colormap(plt.cm.hot_r, 0, .7)
    return ax.imshow(sequence_landscape.T,
        origin="lower",
        cmap=cmap,
    )
    
def panel_sequence_correlations(ax:object, config:object, tag:str, taskname:str):
    from task import correlate_task_sequences    
    import pandas as pd
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tag)
    merged = correlate_task_sequences(tag, config, len(detection_spots))
    
    df = pd.DataFrame.from_dict(merged, orient='index')
    
    df.rename(columns={0: "baseline", 1: "task"}, inplace=True)
    unused = [
        "C0&C2", "C0&C3", "C0&C4",
        "C1&C3", "C1&C4", #"C2&C4",
        "C1&C3&C4",  "C0&C2&C3",  #"C1&C2&C4",      
        "C0&C1&C3", "C0&C1&C4", "C0&C2&C4", "C0&C3&C4",
        "C0&C1&C3&C4", "C0&C2&C3&C4",
         "C0&C1&C3&C4", #"C0&C1&C2&C4",
    ]
    df.drop(unused, inplace=True)
    order = [
        "C0", "C0&C1", "C1", "C0&C1&C2", "C1&C2", "C2",
        "C0&C1&C2&C3", "C1&C2&C3", "C2&C3", "C3",
        "C0&C1&C2&C4", "C1&C2&C4", "C2&C4",
        "C0&C1&C2&C3&C4", "C1&C2&C3&C4", "C2&C3&C4", "C3&C4", "C4"
    ]
    df = df.reindex(order)
    df.rename({
        "C0&C1": r"C0$\rightarrow$C1",
        "C1&C2": r"C1$\rightarrow$C2",
        "C2&C3": r"C2$\rightarrow$L",
        "C0&C1&C2&C3&C4": r"C0$\rightarrow$L&R",
        "C1&C2&C3&C4": r"C1$\rightarrow$L&R",
        "C2&C3&C4": r"C2$\rightarrow$L&R",
        "C0&C1&C2&C3": r"C0$\rightarrow$L",
        "C0&C1&C2&C4": r"C0$\rightarrow$R",
        "C1&C2&C3": r"C1$\rightarrow$L",
        "C1&C2&C4": r"C1$\rightarrow$R",
        "C0&C1&C2": r"C0$\rightarrow$C2",
        "C4": "R",
        "C3": "L",
        "C3&C4": "L&R",
        "C2&C4": r"C2$\rightarrow$R",
    }, inplace=True)
    redundant = [
        "C0", "C1", "C2", "L", "R", r"C1$\rightarrow$C2",
        r"C2$\rightarrow$L&R", r"C1$\rightarrow$L&R",
    ]
    df.drop(redundant, inplace=True)
    df.plot(ax=ax, kind='bar', legend=False)


#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
