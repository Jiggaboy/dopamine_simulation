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
from lib.pickler_class import Pickler
import lib.universal as UNI

from plot.activity import create_image
from plot.lib import plot_patch

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


marker = ["o", "*", "^", "v", "s"]
bs_color = "k"

def main():
    from params import config

    tags_by_seed = config.get_all_tags(seeds="all")
    colors = _get_colors(len(tags_by_seed))

    # from matplotlib import rcParams
    # rcParams['lines.markersize'] = 30
    # config.analysis.sequence.min_samples = 50
    # test(config, tags_by_seed, colors)

    # from matplotlib import rcParams
    # rcParams['lines.markersize'] = 20
    # config.analysis.sequence.min_samples = 75
    # test(config, tags_by_seed, colors)

    test_cluster(config, min_samples=[50])


# def test_cluster(config, min_samples):
#     from plot.lib import activity_3d as a3d
#     seed = 0
#     baseline_tag = config.baseline_tags[seed]


#     # for min_sample in min_samples:
#     for min_sample in reversed(min_samples):
#         config.analysis.sequence.min_samples = min_sample
#         baseline_spikes, baseline_labels = PIC.load_spike_train(baseline_tag, config)
#         # baseline_durations = _get_durations(baseline_spikes[:, 0], baseline_labels, baseline_labels.max())
#         # print(baseline_durations)
#         # longest = np.argmax(baseline_durations)
#         # shortest = np.argmin(baseline_durations)
#         # shortest = np.argwhere(baseline_durations < 150)
#         # print(len(shortest))
#         selection = baseline_labels[np.logical_and(baseline_spikes[:, 0] >= 2100, baseline_spikes[:, 0] <= 2250)]
#         a3d.plot_cluster(baseline_spikes, baseline_labels, force_label=np.unique(selection))
#         # a3d.plot_cluster(baseline_spikes, baseline_labels, force_label=shortest)
#         break
#     return

#     config.analysis.sequence.min_samples = 50
#     baseline_spikes_50, baseline_labels_50 = PIC.load_spike_train(baseline_tag, config)
#     baseline_durations_50 = _get_durations(baseline_spikes_50[:, 0], baseline_labels_50, baseline_labels_50.max())
#     longest_50 = np.argmax(baseline_durations_50)
#     bs50_long_spikes = baseline_spikes_50[baseline_labels_50 == longest_50]


#     config.analysis.sequence.min_samples = 75
#     baseline_spikes_75, baseline_labels_75 = PIC.load_spike_train(baseline_tag, config)
#     baseline_durations_75 = _get_durations(baseline_spikes_75[:, 0], baseline_labels_75, baseline_labels_75.max())
#     longest_75 = np.argmax(baseline_durations_75)
#     bs75_long_spikes = baseline_spikes_75[baseline_labels_75 == longest_75]
#     print(bs75_long_spikes)

#     # find difference in both
#     bs50_long_spikes = np.unique(bs50_long_spikes, axis=1)
#     bs75_long_spikes = np.unique(bs75_long_spikes, axis=1)

#     common = []
#     diff = []
#     for spike in bs50_long_spikes:
#         if np.count_nonzero((spike == bs75_long_spikes).all(-1)):
#             common.append(spike)
#         else:
#             diff.append(spike)

#     common = np.asarray(common)
#     diff = np.asarray(diff)

#     a3d.plot_cluster(diff)



# def test(config, tags_by_seed, colors):
#     for tag in tags_by_seed[0]:
#         _, seed = UNI.split_seed_from_tag(tag)
#         seed = int(seed)
#         if seed in (3, 4):
#             continue
#         baseline_tag = config.get_baseline_tag_from_tag(tag)
#         plot_kwargs = {"edgecolors": "k", "marker": marker[seed]}
#         baseline_spikes, baseline_labels = PIC.load_spike_train(baseline_tag, config)
#         baseline_durations = _get_durations(baseline_spikes[:, 0], baseline_labels, baseline_labels.max())
#         plt.figure(1)
#         plt.scatter(baseline_labels.max(), baseline_durations.mean(), color=bs_color, **plot_kwargs)

#         plt.figure(1)
#         plt.plot(np.sort(baseline_durations)[::-1], label=config.analysis.sequence.min_samples)
#         print(set(sorted(baseline_labels)))
#         break
#     plt.legend()

def main():
    from params import config
    if UNI.yes_no("Plot Differences between NM and baseline?", False):
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

    if UNI.yes_no("Plot sequence count and duration?", True):
        from matplotlib import rcParams
        rcParams['lines.markersize'] = 5
        # config.analysis.sequence.min_samples = 50
        plot_count_and_duration(config)
        # from matplotlib import rcParams
        # rcParams['lines.markersize'] = 15
        # config.analysis.sequence.min_samples = 75
        # plot_count_and_duration(config)
    if UNI.yes_no("Plot difference across sequence counts?", False):
        plot_seq_diff(config)
    plt.show()


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



def plot_count_and_duration(config:object):
    fig, axes = plt.subplots(ncols=3, num="count_and_duration")

    ax_seq_count, ax_duration, ax_sum = axes
    ax_seq_count.set_title("Seq. count")
    ax_seq_count.set_ylabel("# sequences")

    ax_duration.set_title("Avg. Duration")
    ax_duration.set_ylabel("duration [time steps]")

    marker = ["o", "*", "^", "v", "s"]
    fig, axes = plt.subplots(ncols=1, num="test")

    tags_by_seed = config.get_all_tags(seeds="all")
    colors = _get_colors(len(tags_by_seed))
    bs_x = -1
    bs_color = "k"

    for tag in tags_by_seed[0]:
        _, seed = UNI.split_seed_from_tag(tag)
        seed = int(seed)

        baseline_tag = config.get_baseline_tag_from_tag(tag)
        _plot_count_and_duration(baseline_tag, bs_x, config, axes, marker=marker[seed], color=bs_color, lw=3)

    for s, tag_seeds in enumerate(tags_by_seed):
        for tag in tag_seeds:
            _, seed = UNI.split_seed_from_tag(tag)
            seed = int(seed)
            _plot_count_and_duration(tag, s, config, axes, color=colors[s], lw=3)
    ax_duration.legend()


def _plot_count_and_duration(tag:str, x_pos:float, config:object, axes:tuple, **plot_kwargs):
    _, seed = UNI.split_seed_from_tag(tag)
    seed = int(seed)
    if seed == 0:
        plot_kwargs["label"] = tag

    spikes, labels = PIC.load_spike_train(tag, config)
    seq_count = labels.max()
    # axes[0].plot(x_pos, seq_count, **plot_kwargs)

    durations = _get_durations(spikes[:, 0], labels, seq_count)
    durations = list(reversed(sorted(durations)))
    axes.plot(np.arange(len(durations)), durations, **plot_kwargs)
    # vplot = axes[1].violinplot(durations, [x_pos + seed*.1])
    # for b in vplot["bodies"]:
    #     b.set_color(plot_kwargs["color"])
    # axes[1].plot(x_pos, durations.mean(), **plot_kwargs)

    # axes[2].plot(x_pos, durations.sum(), **plot_kwargs)


def plot_count_and_duration(config:object):
    marker = ["o", "*", "^", "v", "s"]
    figname = "duration over sequence count"
    if not plt.fignum_exists(figname):
        fig, ax = plt.subplots(ncols=1, num=figname, figsize=(3, 2.6), tight_layout=True)
        ax.set_title("Activation across conditions")
        ax.set_xlabel("# sequences")
        ax.set_ylabel("Avg. duration")
        ax.set_xticks([100, 115, 130])
        ax.set_yticks([210, 290, 370])
        ax.set_ylim([205, 375])
    else:
        fig = plt.figure(figname)

    tags_by_seed = config.get_all_tags(seeds="all")
    colors = _get_colors(len(tags_by_seed))
    bs_x = -1
    bs_color = "k"
    # from matplotlib import rcParams
    # rcParams['lines.markersize'] = 15

    bs_duration = np.zeros(len(tags_by_seed[0]))
    bs_seq_count = np.zeros(len(tags_by_seed[0]))
    for tag in tags_by_seed[0]:
        _, seed = UNI.split_seed_from_tag(tag)
        seed = int(seed)
        baseline_tag = config.get_baseline_tag_from_tag(tag)
        plot_kwargs = {"marker": "o", "label": baseline_tag, "zorder": 20}
        plot_kwargs["label"] = "baseline"
        # plot_kwargs = {"edgecolors": "k", "marker": marker[seed], "label": baseline_tag, "zorder": 2}
        # if seed == 0:
        #     plot_kwargs["label"] = "baseline"
        # else:
        #     plot_kwargs["label"] = None
        baseline_spikes, baseline_labels = PIC.load_spike_train(baseline_tag, config)
        baseline_durations = _get_durations(baseline_spikes[:, 0], baseline_labels, baseline_labels.max())
        bs_duration[seed] = baseline_durations.mean()
        bs_seq_count[seed] = baseline_labels.max()

        # plt.scatter(baseline_labels.max(), baseline_durations.mean(), color=bs_color, **plot_kwargs)
    plt.errorbar(bs_seq_count.mean(), bs_duration.mean(), xerr=bs_seq_count.std(), yerr=bs_duration.std(), color=bs_color, **plot_kwargs)

    for s, tag_seeds in enumerate(tags_by_seed):
        duration_across_seeds = np.zeros(len(tag_seeds))
        seq_count_across_seeds = np.zeros(len(tag_seeds))
        for tag in tag_seeds:
            _, seed = UNI.split_seed_from_tag(tag)
            seed = int(seed)


            # baseline_tag = config.get_baseline_tag_from_tag(tag)


            spikes, labels = PIC.load_spike_train(tag, config)
            durations = _get_durations(spikes[:, 0], labels, labels.max())
            # baseline_spikes, baseline_labels = PIC.load_spike_train(baseline_tag, config)
            # baseline_durations = _get_durations(baseline_spikes[:, 0], baseline_labels, baseline_labels.max())

            plot_kwargs = {"edgecolors": "k", "marker": marker[seed], }
            plot_kwargs = {"marker": "o", }
            p = UNI.split_percentage_from_tag(tag)
            plot_kwargs["label"] = f"{int(p):+}%"

            # if seed == 0:
            #     p = UNI.split_percentage_from_tag(tag)
            #     plot_kwargs["label"] = f"{int(p):+}%"
            # else:
            #     plot_kwargs["label"] = None

            duration_across_seeds[seed] = durations.mean()
            seq_count_across_seeds[seed] = labels.max()
        plt.errorbar(seq_count_across_seeds.mean(), duration_across_seeds.mean(), xerr=seq_count_across_seeds.std(), yerr=duration_across_seeds.std(), **plot_kwargs)

            # plt.scatter(labels.max(), durations.mean(), color=colors[s], **plot_kwargs)
            # plot_kwargs["label"] = None

            # max_length = max(len(baseline_durations), len(durations))
            # min_length = min(len(baseline_durations), len(durations))
            # durations_long = np.zeros(max_length)
            # baseline_durations_long = np.zeros(max_length)

            # # fill up the shorter array
            # if len(durations) < max_length:
            #     durations_long[:len(durations)] = durations
            # else:
            #     durations_long = durations
            # if len(baseline_durations) < max_length:
            #     baseline_durations_long[:len(baseline_durations)] = baseline_durations
            # else:
            #     baseline_durations_long = baseline_durations


            # diff = np.sort(durations_long)[::-1] - np.sort(baseline_durations_long)[::-1]

            # plt.plot(np.arange(min_length), diff[:min_length], color=colors[s])
            # plt.plot(np.arange(min_length-1, max_length), diff[min_length-1:], ls="--", color=colors[s])
    plt.legend(
        fontsize="small",
         scatteryoffsets=[0.5],
         # frameon=False,
         # loc="upper right",
         labelspacing=.2,
         alignment = "right"
     )




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
