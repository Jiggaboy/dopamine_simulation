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


import lib.pickler as PIC
import lib.universal as UNI

from plot import COLORS
from plot import SequenceConfig
from plot.activity import create_image

MS = SequenceConfig.marker_size
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']



def main():
    from params import config
    tags = config.get_all_tags()
    if tags == []:
        for tag in config.baseline_tags:
            plot_sequence_landscape(tag, config)
    else:
        for tag in tags[:]:
            # plot_sequences_at_location(tag, config, is_baseline=False)
            tag_tmp = config.get_baseline_tag_from_tag(tag)
            plot_sequence_landscape(tag_tmp, config)
            plot_sequence_landscape(tag, config)

    plt.show()


    # if UNI.yes_no("Plot sequence count and duration?"):
    #     plot_count_and_duration(config)
    # if UNI.yes_no("Plot difference across sequence counts?"):
    #     plot_seq_diff(config)

def plot_sequence_landscape(tag, config:object) -> None:
    num = f"sequences_{tag}"
    if plt.fignum_exists(num):
        return
    plt.figure(num)
    spikes, labels = PIC.load_spike_train(tag, config)
    seq_count = np.zeros(shape=(config.rows, config.rows), dtype=int)
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        spike_set = spikes[labels == label]
        spike_set = np.unique(spike_set[:, 1:], axis=0).T
        seq_count[tuple(spike_set)] += 1

    im = create_image(seq_count.T, norm=(0, np.max(seq_count)))
    plt.colorbar(im)

def plot_sequences_at_location(tag:str, config:object, is_baseline:bool):
    from plot.lib import plot_cluster

    tag_tmp = config.get_baseline_tag_from_tag(tag) if is_baseline else tag
    spikes, labels = PIC.load_spike_train(tag_tmp, config)
    plot_cluster(spikes, labels)


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
        # break



def plot_count_and_duration(config:object):
    fig, axes = plt.subplots(ncols=3, num="count_and_duration")

    ax_seq_count, ax_duration, ax_sum = axes
    ax_seq_count.set_title("Seq. count")
    ax_seq_count.set_ylabel("# sequences")

    ax_duration.set_title("Avg. Duration")
    ax_duration.set_ylabel("duration [time steps]")

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
    # ax_duration.legend()

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


if __name__ == "__main__":
    main()
    plt.show()
