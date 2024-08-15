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
from matplotlib import rcParams
import numpy as np


import lib.pickler as PIC
from lib.pickler_class import Pickler
import lib.universal as UNI

from plot.activity import create_image
from plot.lib import plot_patch


marker = ["o", "*", "^", "v", "s"]
bs_color = "k"
rcParams['lines.markersize'] = 5

#===============================================================================
# MAIN
#===============================================================================

def main():
    from params import config
    if UNI.yes_no("Plot Sequence count on location?", False):
        tags = config.get_all_tags()
        # Plots only baseline if no tags are detected
        if tags == []:
            for tag in config.baseline_tags:
                plot_sequence_landscape(tag, config)
        else:
            for tag in tags[:]:
                tag_tmp = config.get_baseline_tag_from_tag(tag)
                plot_sequence_landscape(tag_tmp, config)
                plot_sequence_landscape(tag, config)

    if UNI.yes_no("Plot sequence count and duration?", True):
        plot_count_and_duration(config)

    if UNI.yes_no("Plot difference across sequence counts?", False):
        plot_seq_diff(config)

    if UNI.yes_no("Plot sequence duration over indegree?", False):
        plot_seq_duration_over_indegree(config, feature="duration")
        plot_seq_duration_over_indegree(config, feature="sequence")
    plt.show()


#===============================================================================
# SEQUENCE LANDSCAPE & Sequence difference
#===============================================================================

def plot_sequence_landscape(tag, config:object, plot_diff:bool=False, save:bool=True) -> None:
    num = f"sequences_{tag}" if not plot_diff else f"sequences_diff_{tag}"
    name = UNI.name_from_tag(tag)
    radius = UNI.radius_from_tag(tag)
    if plt.fignum_exists(num):
        return
    fig = plt.figure(num)
    # spikes, labels = PIC.load_spike_train(tag, config)
    seq_count = _get_sequence_landscape(tag, config)
    if config.get_baseline_tag_from_tag(tag) == tag:
        im = create_image(seq_count.T, norm=(0, np.max(seq_count)))
        plt.colorbar(im)
    else:
        seq_count_bs = _get_sequence_landscape(config.get_baseline_tag_from_tag(tag), config)
        if plot_diff:
            seq_diff = seq_count - seq_count_bs
            _max = np.max(np.abs(seq_diff))
            im = create_image(seq_diff.T, norm=(-_max, _max), cmap="seismic")
        else:
            im = create_image(seq_count.T, norm=(0, seq_count.max()))
        plt.colorbar(im)
    if name in config.center_range.keys():
        plot_patch(config.center_range[name], float(radius), config.rows)

    if save:
        pickler = Pickler(config)
        pickler.save_figure(num, fig)


def _get_sequence_landscape(tag:str, config:object):
    spikes, labels = PIC.load_spike_train(tag, config)
    seq_count = np.zeros(shape=(config.rows, config.rows), dtype=int)
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        spike_set = spikes[labels == label]
        spike_set = np.unique(spike_set[:, 1:], axis=0).T
        seq_count[tuple(spike_set)] += 1
    return seq_count

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


#===============================================================================
# COUNT AND DURATION VS INDEGREE
#===============================================================================
def plot_seq_duration_over_indegree(config:object, feature:str=None) -> None:
    figname = f"{feature} over indegree"
    if not plt.fignum_exists(figname):
        fig, axes = plt.subplots(
            ncols=len(config.PERCENTAGES),
            num=figname,
            figsize=(4, 2.6),
            tight_layout=True,
            sharey=True,
        )
        fig.suptitle("Activation across conditions")
        for ax in axes:
            if ax == axes[0]:
                ax.set_ylabel("Difference in avg. duration")

            ax.set_xlabel("Median Patch Indegree")
            ax.axhline(c="k", lw=2)
    else:
        # Never reaches here? Axes is not defined
        raise LookupError
        fig = plt.figure(figname)

    for i, p in enumerate(config.PERCENTAGES):
        tags_by_seed = config.get_all_tags(seeds="all", weight_change=[p])
        plt.sca(axes[i])
        plt.title(f"{int(100*p):+}%")

        for s, tag_seeds in enumerate(tags_by_seed):
            _plot_feature_vs_indegree(config, tag_seeds, feature=feature)

    plt.legend(
        fontsize="small",
          scatteryoffsets=[0.5],
          labelspacing=.2,
      )



#===============================================================================
# COUNT AND DURATION
#===============================================================================

def plot_count_and_duration(config:object):
    figname = "duration over sequence count"
    if not plt.fignum_exists(figname):
        fig, axes = plt.subplots(
            ncols=len(config.PERCENTAGES),
            num=figname,
            figsize=(4, 2.6),
            tight_layout=True,
            sharey=True,
        )
        fig.suptitle("Activation across conditions")
        for ax in axes:
            if ax == axes[0]:
                ax.set_ylabel("Avg. duration")
                ax.set_yticks([220, 290, 360])
                ax.set_ylim([205, 375])

            ax.set_xlabel("# sequences")
            ax.set_xticks([75, 90, 105])
            ax.set_xlim([70, 110])
    else:
        fig = plt.figure(figname)

    for i, p in enumerate(config.PERCENTAGES):
        tags_by_seed = config.get_all_tags(seeds="all", weight_change=[p])
        plt.sca(axes[i])
        plt.title(f"{int(100*p):+}%")


        plot_kwargs = {"marker": "o", "label": "baseline", "zorder": 20, "markerfacecolor": bs_color} #, "color": bs_color
        _plot_count_vs_duration(config, tags_by_seed[0], is_baseline=True, **plot_kwargs)
        for s, tag_seeds in enumerate(tags_by_seed):
            p = UNI.split_percentage_from_tag(tag_seeds[0])
            plot_kwargs = {
                "marker": "o",
                # "label": f"{int(p):+}%"
            }
            _plot_count_vs_duration(config, tag_seeds, **plot_kwargs)


    plt.legend(
        fontsize="small",
          scatteryoffsets=[0.5],
          labelspacing=.2,
      )


def _plot_count_vs_duration(config:object, tag_across_seed:list, is_baseline:bool=False, **plot_kwargs) -> None:
    duration = np.zeros(len(tag_across_seed))
    sequence_count = np.zeros(len(tag_across_seed))
    for seed, tag in enumerate(tag_across_seed):
        if is_baseline:
            tag = config.get_baseline_tag_from_tag(tag)

        spikes, labels = PIC.load_spike_train(tag, config)
        durations = _get_durations(spikes[:, 0], labels, labels.max())

        duration[seed] = durations.mean()
        sequence_count[seed] = labels.max()

    if is_baseline:
        color = "k"
    else:
        indegree, color = get_indegree(config, tag_across_seed)
    plt.errorbar(sequence_count.mean(), duration.mean(),
                 xerr=sequence_count.std(), yerr=duration.std(),
                 color = color, **plot_kwargs)
    return


def _plot_feature_vs_indegree(config:object, tag_across_seed:list, feature:str=None, **plot_kwargs) -> None:
    duration = np.zeros(len(tag_across_seed))
    sequence_count = np.zeros(len(tag_across_seed))
    for seed, tag in enumerate(tag_across_seed):

        bs_tag = config.get_baseline_tag_from_tag(tag)

        spikes, labels = PIC.load_spike_train(tag, config)
        durations = _get_durations(spikes[:, 0], labels, labels.max())

        bs_spikes, bs_labels = PIC.load_spike_train(bs_tag, config)
        bs_durations = _get_durations(bs_spikes[:, 0], bs_labels, bs_labels.max())

        duration[seed] = durations.mean() - bs_durations.mean()
        sequence_count[seed] = labels.max() - bs_labels.max()

    indegree, color = get_indegree(config, tag_across_seed)

    feature = feature if feature is not None else "duration"
    if "duration" in feature.lower():
        feature = duration
    else:
        feature = sequence_count

    plt.errorbar(indegree, feature.mean(), yerr=feature.std(), marker="_", color=color)

    # plt.plot(indegree, feature.mean(), color=color, ls="None", marker="o", markersize=10)
    # for d in feature:
    #     plt.plot(indegree, d, color=color, ls="None", marker="+")

def _get_durations(times:np.ndarray, labels:np.ndarray, seq_count:int) -> np.ndarray:
    durations = np.zeros(seq_count)
    for l in range(seq_count):
        idx = labels == l
        durations[l] = times[idx].max() - times[idx].min()
    return durations


def get_indegree(config:object, tags:list):
    import lib.dopamine as DOP

    name = UNI.name_from_tag(tags[0])
    center = config.center_range[name]

    # coordinates = UNI.get_coordinates(config.rows)

    radius = UNI.radius_from_tag(tags[0])
    patch = DOP.circular_patch(config.rows, center, float(radius))
    patch = patch.reshape((config.rows, config.rows))


    from lib.connectivitymatrix import ConnectivityMatrix
    conn = ConnectivityMatrix(config).load()

    _, indegree = conn.degree(conn.connections[:config.rows**2, :config.rows**2])
    patch_indegree = indegree[patch].max() * config.synapse.weight
    patch_indegree = np.median(indegree[patch]) * config.synapse.weight

    degree_cmap = plt.cm.jet
    min_degree = 550
    max_degree = 950
    patch_indegree = min_degree if patch_indegree < min_degree else patch_indegree
    patch_indegree = max_degree if patch_indegree > max_degree else patch_indegree
    color = degree_cmap((patch_indegree - min_degree) / (max_degree - min_degree))
    return patch_indegree, color

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
