#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Plots the number of detected sequences in a network for a particular configuration.

Description:


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
import itertools


import lib.pickler as PIC
import lib.universal as UNI

from plot import COLORS
from plot import SequenceConfig

MS = SequenceConfig.marker_size
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']



def main():
    from params import config

    if UNI.yes_no("Plot detected sequences?"):
        plot_db_sequences(config, config.get_all_tags("repeat-early"))

    if UNI.yes_no("Plot sequence count and duration?"):
        plot_count_and_duration(config)
    if UNI.yes_no("Plot difference across sequence counts?"):
        plot_seq_diff(config)


def _get_markup(pre:int, post:int) -> dict:
    # Fix the color for any combination of pre and post
    markup = {}
    markup["color"] = color_cycle[pre + post - 1]
    markup["marker"] = SequenceConfig.marker.default if pre < post else SequenceConfig.marker.alternative
    return markup


def plot_db_sequences(config, tags:list):
    """Plots the number of detected sequences using different methods (thresholding, mean thresholding, clustering)."""
    tags = UNI.make_iterable(tags)
    for tag in tags:
        fig, ax = plt.subplots(num=tag)
        plot_sequences(config, tag, axis=ax)
        PIC.save_figure(f"seq_db_{tag}", fig, config.sub_dir)


def plot_sequences(config:object, tag:str, axis, **plot_kwargs):
    centers =config.analysis.dbscan_controls.detection_spots_by_tag(tag)
    sequence_at_center = PIC.load_sequence_at_center(tag, centers, config)

    _, seed = UNI.split_seed_from_tag(tag)
    tag_baseline = config.baseline_tags[int(seed)]

    sequence_at_center_baseline = PIC.load_sequence_at_center(tag_baseline, centers, config)

    ref = 0.
    distance = .4
    x_margin = .05
    handles = []
    for idx, (center, color) in enumerate(zip(centers, COLORS)):
        params = {"c": color, "axis": axis}
        _ = scatter(ref, np.count_nonzero(sequence_at_center_baseline, axis=0)[idx], **params, **plot_kwargs)
        handle = scatter(distance, np.count_nonzero(sequence_at_center, axis=0)[idx], **params, **plot_kwargs)
        handles.append(handle)

    shared = np.count_nonzero(sequence_at_center_baseline.all(axis=1))
    scatter(ref, shared, axis=axis, **plot_kwargs)
    shared = np.count_nonzero(sequence_at_center.all(axis=1))
    scatter(distance, shared, axis=axis, **plot_kwargs)

    axis.set_ylabel("# sequences")
    axis.set_xticks([ref, distance], labels=["w/o patch", "w/ patch"])
    axis.set_xlim([ref - x_margin, distance + x_margin])
    axis.set_ylim(bottom=-1)
    axis.legend(handles=handles, labels=centers)
    plt.tight_layout()


def plot_seq_diff(config:object, cmap:str="seismic"):
    from plot.activity import create_image
    tags = config.get_all_tags()

    for tag in tags:
        baseline_tag = config.get_baseline_tag_from_tag(tag)
        baseline_spikes, baseline_labels = PIC.load_spike_train(baseline_tag, config)
        patch_spikes, patch_labels = PIC.load_spike_train(tag, config)

        coordinates_bs = baseline_spikes[:, 1:]
        H_bs, _, _ = np.histogram2d(*coordinates_bs.T, bins=np.arange(-0.5, config.rows))

        coordinates = patch_spikes[:, 1:]
        H, _, _ = np.histogram2d(*coordinates.T, bins=np.arange(-0.5, config.rows))

        plt.figure(tag)
        plt.title(f"{tag} with {len(set(patch_labels))} and {len(set(baseline_labels))} Sequences")
        im = create_image(H.T - H_bs.T, norm=(-200, 200), cmap=cmap)
        plt.colorbar(im)
        break



def plot_count_and_duration(config:object):
    fig, axes = plt.subplots(ncols=3, num="count_and_duration")

    marker = ["o", "*", "^", "v"]

    tags_by_seed = config.get_all_tags(seeds="all")
    colors = _get_colors(len(tags_by_seed))
    bs_x = -1
    bs_color = "k"

    for tag in tags_by_seed[0]:
        _, seed = UNI.split_seed_from_tag(tag)
        seed = int(seed)

        baseline_tag = config.get_baseline_tag_from_tag(tag)
        _plot_count_and_duration(baseline_tag, bs_x, config, axes, marker=marker[seed], color=bs_color)

    for s, tag_seeds in enumerate(tags_by_seed):
        for tag in tag_seeds:
            _, seed = UNI.split_seed_from_tag(tag)
            seed = int(seed)
            _plot_count_and_duration(tag, s, config, axes, marker=marker[seed], color=colors[s])


def _plot_count_and_duration(tag:str, x_pos:float, config:object, axes:tuple, **plot_kwargs):
    _, seed = UNI.split_seed_from_tag(tag)
    seed = int(seed)

    spikes, labels = PIC.load_spike_train(tag, config)
    seq_count = labels.max()
    axes[0].plot(x_pos, seq_count, label=tag, **plot_kwargs)

    durations = _get_durations(spikes[:, 0], labels, seq_count)
    axes[1].plot(x_pos, durations.mean(), label=tag, **plot_kwargs)

    axes[2].plot(x_pos, durations.sum(), label=tag, **plot_kwargs)


def _get_colors(number:int, cmamp:str="gist_rainbow"):
    cm = plt.get_cmap(cmamp)
    return [cm(1. * i / number) for i in range(number)]


def _get_durations(times:np.ndarray, labels:np.ndarray, seq_count:int) -> np.ndarray:
    durations = np.zeros(seq_count)
    for l in range(seq_count):
        idx = labels == l
        durations[l] = times[idx].max() - times[idx].min()
    return durations




########################################################################################################################
##### Correlation Analysis #############################################################################################
########################################################################################################################

def scatter_sequence_at_location(sequence_at_center:np.ndarray, center:list, **plt_kwargs) -> None:
    """
    Plots the sequences that cross at the centers.
    Used for sequence correlation analysis.

    Parameters
    ----------
    sequence_at_center : np.ndarray
        Boolean array with shape (n, c), with n being the number of detected sequences and c the number of centers.
    center : list
        Analysis locations.

    Returns
    -------
    None

    """
    sequences_with_id = sequence_at_center * np.arange(1, len(center)+1)
    plt.figure(**plt_kwargs)
    plt.plot(sequences_with_id, marker="*", ls="None", label=[str(c) for c in center])
    plt.legend()


def imshow_correlations(correlations:np.ndarray, is_baseline:bool=True, tag:str=None, ax:object=None) -> None:
    if ax is None:
        plt.figure(f"correlations_{tag}")
    plot_obj = ax if ax is not None else plt
    title_setter = ax.set_title if ax is not None else plt.title
    colorbar_setter = ax.figure.colorbar if ax is not None else plt.colorbar

    title_tag = "Baseline" if is_baseline else "Patch"
    title_setter(f"{title_tag} - {tag}")
    im = plot_obj.imshow(correlations, vmin=0, vmax=1, cmap="jet")
    colorbar_setter(im)


def imshow_correlation_difference(correlation_diff:np.ndarray, ax:object=None) -> None:
    ax.set_title("Difference in correlations (Patch - Baseline)")
    im = ax.imshow(correlation_diff, vmin=-.5, vmax=.5, cmap="seismic")
    ax.figure.colorbar(im)

########################################################################################################################


def scatter(x:np.ndarray, data:np.ndarray, axis:object=None, **kwargs):
    # TODO: What is the benefit (except having a defaults?)
    ax = axis if axis is not None else plt
    plot_to_scatter = {"ls": "--", "marker": "o", "markersize": MS}
    plot_to_scatter.update(kwargs)
    try:
        line, = ax.plot(np.full(shape=len(data), fill_value=x), data, **plot_to_scatter)
    except TypeError:
        line, = ax.plot(x, data, **plot_to_scatter)
    return line



if __name__ == "__main__":
    main()
    plt.show()
