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
import matplotlib as mpl
from matplotlib import rcParams
import numpy as np

from plot.constants import *
from lib.neuralhdf5 import NeuralHdf5, default_filename
import lib.universal as UNI
import lib.pickler as PIC
import lib.dopamine as DOP
from analysis.sequence_correlation import SequenceCorrelator
from plot.lib.frame import create_image
from plot.lib import add_colorbar, plot_patch_from_tag, plot_patch, get_color, add_colorbar_from_im
from plot.lib import remove_spines_and_ticks, remove_topright_spines, add_topright_spines

from params import GateConfig
config = GateConfig() # Use a specific one here!
#===============================================================================
# CONSTANTS
#===============================================================================
figsize = (17.6*cm, 12*cm)


legend_kwargs = {"ncol": 2}

filename = "cooperation"

percent = .1
radius = 6
name = "gate-left"
tag_cooperation = config.get_all_tags(name, radius=radius, weight_change=percent, seeds=4)[0]
coop_sequence_idx = 26
tag_competition = config.baseline_tag(3)
B1_idx = 3
B2_idx = 4

vline_style = {"ls": "--"}
time_cmap = plt.cm.jet
cmap = mpl.colormaps.get_cmap(time_cmap)
cmap.set_bad(color='white')
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    correlator = SequenceCorrelator(config)
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tag_cooperation)
    neurons_per_detection_spot = np.asarray([DOP.circular_patch(config.rows, center, radius=2) for center in detection_spots])

    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=5, width_ratios=(1, .45, 1.1, 0.5, .8))
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.92,
        wspace=0.08,
        hspace=0.4,
    )

    ax_comp = fig.add_subplot(gs[0, -3])
    ax_comp.set_title("Spikes over time")
    add_topright_spines(ax_comp)
    ax_comp.set_xlabel("X", **xylabel_style)
    ax_comp.set_ylabel("Y", **xylabel_style)
    shifted_ticks = ((10, 30, 49, 50, 70, 90), (60, 80, 99, " ", 20, 40))
    normal_ticks  = [20, 40, 60]
    xlim = (15, 65)
    ylim = (25, 75)
    ax_comp.set_yticks(*shifted_ticks)
    ax_comp.set_xticks(normal_ticks)
    ax_comp.set_xlim(xlim)
    ax_comp.set_ylim(ylim)
    
    ax_coop = fig.add_subplot(gs[1, -3])
    ax_coop.set_title("Spikes over time")
    add_topright_spines(ax_coop)
    ax_coop.set_xlabel("X", **xylabel_style)
    ax_coop.set_ylabel("Y", **xylabel_style)
    ax_coop.set_yticks(*shifted_ticks)
    ax_coop.set_xticks(normal_ticks)
    ax_coop.set_xlim(xlim)
    ax_coop.set_ylim(ylim)
    
    ylim = (0, 85)
    seq_ticks  = np.arange(0, 75+1, 25)
    ax_comp_neurons = fig.add_subplot(gs[0, -1])
    remove_topright_spines(ax_comp_neurons)
    ax_comp_neurons.set_title("Active Neurons")
    ax_comp_neurons.set_xlabel("Time [ms]", **xylabel_style)
    ax_comp_neurons.set_ylabel("# of activated neurons", **xylabel_style)
    ax_comp_neurons.set_yticks(seq_ticks)
    ax_comp_neurons.set_ylim(ylim)
    
    
    ylim = (0, 95)
    ax_coop_neurons = fig.add_subplot(gs[1, -1])
    remove_topright_spines(ax_coop_neurons)
    ax_coop_neurons.set_title("Active Neurons")
    ax_coop_neurons.set_xlabel("Time [ms]", **xylabel_style)
    ax_coop_neurons.set_ylabel("# of activated neurons", **xylabel_style)
    ax_coop_neurons.set_yticks(seq_ticks)
    ax_coop_neurons.set_ylim(ylim)
    
    ##### TOP: COMPETITION ##############################
    spikes, labels = PIC.load_spike_train(tag_competition, config=config)
    comp_spikes = spikes[np.logical_or(labels == 3, labels == 4)]
    sequence_on_location = get_sequence_on_location(comp_spikes, (config.rows, config.rows))
    
    sequence_on_location_tmp = np.roll(sequence_on_location, 50, axis=1)
    
    t_min, t_max = comp_spikes[:, 0].min(), comp_spikes[:, 0].max()
    t_min, t_max = 100, 325
    create_image(sequence_on_location_tmp.T, axis=ax_comp, 
        norm=(t_min, t_max),
        cmap=cmap,
    )
    cbar = add_colorbar(ax_comp, norm=(t_min, t_max), cmap=cmap)
    cbar.set_label("Time [ms]", rotation=270, labelpad=12)
    cbar.set_ticks([150, 300])
    
    
    # Load data and filter by columns (the times and the indicator)
    df_sequence_at_center = correlator.detect_sequence_at_center(tag_competition, center=detection_spots)
    # C0_time, C1_time, C2_time --> Times at which the sequences crosses the detection spot
    time_column_mask = df_sequence_at_center.columns.str.contains('^C\d_time$')
    time_sequence_at_center = df_sequence_at_center.iloc[:, time_column_mask]
    B1_crossing = df_sequence_at_center[df_sequence_at_center["sequence id"] == B1_idx]["C0_time"].item()
    ax_comp_neurons.axvline(B1_crossing, label=r"$\circ$B1", **vline_style, c="tab:blue")
    B2_crossing = df_sequence_at_center[df_sequence_at_center["sequence id"] == B2_idx]["C1_time"].item()
    ax_comp_neurons.axvline(B2_crossing, label=r"$\circ$B2", **vline_style, c="tab:orange")
    M_crossing = df_sequence_at_center[df_sequence_at_center["sequence id"] == B2_idx]["C2_time"].item()
    ax_comp_neurons.axvline(M_crossing, label=r"$\circ$M", **vline_style, c="tab:purple")
    ax_comp_neurons.axvline(260, label="Comp.", **vline_style, c="tab:red")
    
    B1_spikes = spikes[labels == B1_idx]
    H, edges = hist_spike_over_time(B1_spikes)
    edge_center = (edges[1:] + edges[:-1]) / 2
    ax_comp_neurons.plot(edge_center, H)
    
    B2_spikes = spikes[labels == B2_idx]
    H, edges = hist_spike_over_time(B2_spikes)
    edge_center = (edges[1:] + edges[:-1]) / 2
    ax_comp_neurons.plot(edge_center, H)
    
    B2_spikes = spikes[labels == B2_idx]
    H, edges = hist_spike_over_time(B2_spikes[B2_spikes[:, 0] > M_crossing])
    edge_center = (edges[1:] + edges[:-1]) / 2
    ax_comp_neurons.plot(edge_center, H, c="tab:purple")
    
    ax_comp_neurons.set_xticks((150, 300))
    ax_comp_neurons.set_xlim(t_min, t_max)
    ax_comp_neurons.legend(loc="upper center", ncols=2)
    
    ##### BOTTTOM: COOPERATION ##############################
    spikes, labels = PIC.load_spike_train(tag_cooperation, config=config)
    coop_spikes = spikes[labels == coop_sequence_idx]

    sequence_on_location = get_sequence_on_location(coop_spikes, (config.rows, config.rows))
    
    sequence_on_location_tmp = np.roll(sequence_on_location, 50, axis=1)
    
    t_min, t_max = coop_spikes[:, 0].min(), coop_spikes[:, 0].max()
    t_min, t_max = 1120, 1310
    im = create_image(sequence_on_location_tmp.T, axis=ax_coop,
        norm=(t_min, t_max),
        cmap=cmap,
    )
    cbar = add_colorbar_from_im(ax_coop, im)
    cbar.set_label("Time [ms]", rotation=270, labelpad=8)
    # cbar.set_ticks([300, 400])
    
    ec, alpha = get_color(percent)
    kwargs = {"ec": ec, "alpha": alpha}
    roll_offset = np.asarray((0, 50))
    center = config.center_range[name] + roll_offset
    plot_patch(center, radius, width=config.rows, axis=ax_coop, **kwargs)
    for ds in config.analysis.dbscan_controls.detection_spots:
        ds_name, spots = ds
        if ds_name == name:
            for spot in spots:
                plot_patch(spot + roll_offset, radius=2., width=config.rows, axis=ax_coop, lw=1, add_outline=False)
                plot_patch(spot + roll_offset, radius=2., width=config.rows, axis=ax_comp, lw=1, add_outline=False)
    
    
    

    # Load data and filter by columns (the times and the indicator)
    df_sequence_at_center = correlator.detect_sequence_at_center(tag_cooperation, center=detection_spots)
    # C0_time, C1_time, C2_time --> Times at which the sequences crosses the detection spot
    time_column_mask = df_sequence_at_center.columns.str.contains('^C\d_time$')
    time_sequence_at_center = df_sequence_at_center.iloc[:, time_column_mask]
    B1_crossing = df_sequence_at_center[df_sequence_at_center["sequence id"] == coop_sequence_idx]["C0_time"].item()
    ax_coop_neurons.axvline(B1_crossing, label=r"$\circ$B1", **vline_style, c="tab:blue")
    B2_crossing = df_sequence_at_center[df_sequence_at_center["sequence id"] == coop_sequence_idx]["C1_time"].item()
    ax_coop_neurons.axvline(B2_crossing, label=r"$\circ$B2", **vline_style, c="tab:orange")
    M_crossing = df_sequence_at_center[df_sequence_at_center["sequence id"] == coop_sequence_idx]["C2_time"].item()
    ax_coop_neurons.axvline(M_crossing, label=r"$\circ$M", **vline_style, c="tab:purple")
        
    from figure_generator.cooperativity import find_merging_time_point
    # t_min, t_max = coop_spikes[:, 0].min(), coop_spikes[:, 0].max()
    cluster_spikes, cluster_labels, merge_idx = find_merging_time_point(coop_spikes, t_min, t_max)
    ax_coop_neurons.axvline(coop_spikes[merge_idx, 0], color="tab:green", label="Coop.", **vline_style)

    # Analyze the individual clusters
    for i in range(len(set(cluster_labels))):
        # Cluster the number of active neurons per time step
        tmp_spikes = cluster_spikes[cluster_labels == i]
        # Check to which branch the cluster belongs
        # Assumption: At least one spike will be detected at the center of the detection spot
        # tmp_spikes[:, 1:] == # of all coordinates, is any in detection spot
        time_buffer = 150
        for d, detection_spot in enumerate(detection_spots):
            time_crossing = time_sequence_at_center.iloc[coop_sequence_idx, d]
            time_idx = np.logical_and(tmp_spikes[:, 0] > (time_crossing-time_buffer), tmp_spikes[:, 0] < (time_crossing+time_crossing))
            crossed_spot = (tmp_spikes[time_idx, 1:][:, np.newaxis] == config.coordinates[neurons_per_detection_spot[d]]).all(axis=-1).any()
            if crossed_spot:
                break
            else:
                d = None
        
        H, edges = hist_spike_over_time(tmp_spikes)
        edge_center = (edges[1:] + edges[:-1]) / 2
        ax_coop_neurons.plot(edge_center, H)
    
    tmp_spikes = coop_spikes[merge_idx:]
    H, edges = hist_spike_over_time(tmp_spikes)
    edge_center = (edges[1:] + edges[:-1]) / 2
    ax_coop_neurons.plot(edge_center, H, c="tab:purple")
    


    ax_coop_neurons.set_xticks((1100, 1200, 1300))
    ax_coop_neurons.set_xlim(t_min, t_max)
    ax_coop_neurons.legend(loc="upper center", ncols=2)
    
    PIC.save_figure(filename, fig, transparent="True")
#===============================================================================
# METHODS
#===============================================================================
def get_sequence_on_location(spikes, shape:tuple, **kwargs):
    imspikes = np.zeros(shape)
    # sort by coordinates and then the time
    spikes_tmp = spikes[np.lexsort((-spikes[:,0], spikes[:,2], spikes[:,1]))]

    # keep first occurrence of each (b, c)
    _, idx = np.unique(spikes_tmp[:,1:3], axis=0, return_index=True)
    spikes_tmp = spikes_tmp[idx]
    
    imspikes[spikes_tmp[:, 1], spikes_tmp[:, 2]] = spikes_tmp[:, 0]
    imspikes[imspikes == 0] = np.nan
    return imspikes
    

def hist_spike_over_time(spikes:np.ndarray) -> tuple:
    # 0 is time column
    t_min = spikes[:, 0].min()
    t_max = spikes[:, 0].max()
    H, edges = np.histogram(spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5))
    return H, edges
#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
