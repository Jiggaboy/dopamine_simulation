#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:20:21 2022

@author: hauke
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools

from lib import SequenceCounter

import lib.pickler as PIC
import universal as UNI
from analysis import sequence_correlation as SC


from plot.lib import image_slider_1d
from plot import KTH_GREEN, KTH_PINK, KTH_GREY

MS = 8
DISTANCE_BETWEEN_SCATTERS = 0.1
colors = KTH_GREEN, KTH_PINK, KTH_GREY
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


BS_SEQ_FIGSIZE = (12, 6)

def main():
    from params import PerlinConfig, StarterConfig, LowDriveConfig

    cf = LowDriveConfig()

    # all_tags = cf.get_all_tags("out-activator")
    all_tags = cf.get_all_tags("starter-CL")
    all_tags = cf.get_all_tags("gate-bottom")
    all_tags = cf.get_all_tags()
    all_tags_seeds = cf.get_all_tags(seeds="all")

    # for patch in all_tags_seeds:
    #     sequence = PIC.load_db_sequence(patch[0], sub_directory=config.sub_dir)
    #     full_sequence_bs = np.zeros((len(patch), len(sequence.center), sequence.baseline[0].size))
    #     full_sequence_patch = full_sequence_bs.copy()

    #     for tag_idx, tag in enumerate(patch):
    #         sequence = PIC.load_db_sequence(tag, sub_directory=config.sub_dir)
    #         for center_idx, center in enumerate(sequence.center):
    #             full_sequence_bs[tag_idx, center_idx] = sequence.baseline[center_idx]
    #             full_sequence_patch[tag_idx, center_idx] = sequence.patch[center_idx]
    #     sc = SequenceCounter(None, sequence.center)
    #     sc.baseline = full_sequence_bs.mean(axis=0)
    #     sc.patch = full_sequence_patch.mean(axis=0)
    #     sc.baseline_avg = full_sequence_bs.mean(axis=0).mean(axis=1)
    #     sc.patch_avg = full_sequence_patch.mean(axis=0).mean(axis=1)
    #     fig, ax = plt.subplots(num=patch[0], figsize=(2, 3))
    #     ax.set_title("No. of sequences")
    #     plot_sequences(config, patch[0], sequence=sc, axis=ax)
    #     plt.tight_layout()
    #     PIC.save_figure(f"seq_avg_db_{tag}", fig, cf.sub_dir)
    # plt.show()
    # quit()
    # plot_baseline_sequences(config=cf)


    def plot_detailed_correlations(pre, post, id_):
        correlator = SC.SequenceCorrelator(None)
        fig, (ax1, ax2) = plt.subplots(2, num=id_)
        for s in (pre, post):
            train, t_axis, kernel = correlator._convolve_gauss_kernel(s)
            ax1.plot(t_axis, train)
        correlation_normalized, time_axis_correlation = correlator._correlate_sequence_times(pre, post)
        ax2.plot(time_axis_correlation, correlation_normalized)
        ax2.set_xlim(-50, 50)


    def plot_sequence_correlations(config:object, tags:list, add_detailed_plot:bool=False):
        plot_shift = 0
        dshift = .1


        name, seed = UNI.split_seed_from_tag(tags[0])
        print(name, seed)
        fig, _ = plt.subplots(2, num=f"correlations_of_{name}")

        for idx, tag in enumerate(tags):
            sequence = PIC.load_db_cluster_sequence(tag, sub_directory=config.sub_dir)
            len_center = len(sequence.center)
            for pre, post in itertools.permutations(range(len_center), 2):
                _, seed = UNI.split_seed_from_tag(tag)
                c = color_cycle[int(seed)]
                c = color_cycle[pre + post - 1]

                if add_detailed_plot:
                    # TODO
                    print(pre, post)
                    if idx == 0:
                        plot_detailed_correlations(sequence.baseline_times[pre], sequence.baseline_times[post], f"{sequence.center[pre]}_{sequence.center[post]}")

                for i, (corr, spike_times) in enumerate(zip((sequence.correlations_baseline, sequence.correlations_patch), (sequence.baseline_times, sequence.patch_times))):
                    time, correlation = corr[pre, post]

                    max_corr_idx = correlation.argmax()
                    time_lag = time[max_corr_idx]
                    if time_lag > 25 or time_lag < -25:
                        # continue
                        pass
                    marker = "o" if pre < post else "*"
                    fig.axes[0].plot(i+plot_shift, time_lag, c, marker=marker, label=f"{sequence.center[pre]} - {sequence.center[post]}")


                    try:
                        transmission_fraction = np.max(correlation) / spike_times[pre].size
                    except ZeroDivisionError:
                        transmission_fraction = 0
                    fig.axes[1].plot(i+plot_shift, transmission_fraction, c, marker=marker)
                    fig.axes[1].set_ylim(0, 1)
            plot_shift += dshift
            fig.axes[0].legend()

    for tags in all_tags_seeds:
        plot_sequence_correlations(cf, tags, add_detailed_plot=True)
        # break

    # for tag in all_tags:
    #     sequence = PIC.load_db_cluster_sequence(tag, sub_directory=cf.sub_dir)
    #     len_center = len(sequence.center)
    #     for pre, post in itertools.permutations(range(len_center), 2):
    #         fig, _ = plt.subplots(5, num=f"correlations_of_{tag}_{pre}{post}")
    #         # Spike train
    #         correlator = SC.SequenceCorrelator(cf)
    #         for p in (pre, post):
    #             time_series, time_axis, _ = correlator._convolve_gauss_kernel(sequence.baseline_times[p])
    #             fig.axes[0].plot(time_axis, time_series)
    #             time_series, time_axis, _ = correlator._convolve_gauss_kernel(sequence.patch_times[p])
    #             fig.axes[1].plot(time_axis, time_series)
    #         # plot correlations
    #         time, correlation = sequence.correlations_baseline[pre, post]
    #         fig.axes[2].plot(time, correlation)
    #         # plot correlations
    #         time, correlation = sequence.correlations_patch[pre, post]
    #         fig.axes[2].plot(time, correlation)

    #         # get the time lag
    #         time, correlation = sequence.correlations_baseline[pre, post]
    #         max_corr_idx = correlation.argmax()
    #         time_lag = time[max_corr_idx]
    #         fig.axes[3].plot(time_lag, marker="*")

    #         # get the fraction

    #         try:
    #             transmission_fraction = np.max(correlation) / sequence.baseline_times[pre].size
    #         except ZeroDivisionError:
    #             transmission_fraction = 0

    #         fig.axes[4].plot(transmission_fraction, marker="*")

    #         # get the time lag
    #         time, correlation = sequence.correlations_patch[pre, post]
    #         max_corr_idx = correlation.argmax()
    #         time_lag = time[max_corr_idx]
    #         fig.axes[3].plot(1, time_lag, marker="*")

    #         # get the fraction

    #         try:
    #             transmission_fraction = np.max(correlation) / sequence.patch_times[pre].size
    #         except ZeroDivisionError:
    #             transmission_fraction = 0
    #         fig.axes[4].plot(1, transmission_fraction, marker="*")




    # plot_db_sequences(cf, all_tags)
    plt.show()
    return

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


def plot_db_sequences(config, tags:list):
    tags = UNI.make_iterable(tags)

    for tag in tags:
        fig, ax = plt.subplots(num=tag, figsize=(4, 3))
        plot_sequences(config, tag, load_method=PIC.load_db_sequence, axis=ax)
        plot_sequences(config, tag, load_method=PIC.load_db_sequence, axis=ax, average_only=True, ls="-")
        plot_sequences(config, tag, load_method=PIC.load_db_cluster_sequence, axis=ax, marker="*", average_only=True, ls="--")
        PIC.save_figure(f"seq_db_{tag}", fig, config.sub_dir)



def plot_sequences(config:object, tag:str, axis, load_method=None, sequence=None, **plot_kwargs):
    sequence = load_method(tag, sub_directory=config.sub_dir) if sequence is None else sequence
    handles = []
    for idx, (center, c) in enumerate(zip(sequence.center, colors)):
        ####################################################################################################################################################################################################################################################################
        handle = scatter_baseline_patch(idx * DISTANCE_BETWEEN_SCATTERS, sequence, idx, distance=.4, c=c, axis=axis, **plot_kwargs)
        handles.append(handle)
    axis.set_ylabel("# sequences")
    axis.set_xticks([.1, .6], labels=["w/o patch", "w/ patch"])
    axis.set_xlim([-.05, .75])
    #axis.legend(handles=handles, labels=sequence.center)


#def update_sequence(sequence, axis, idx):
#    for idx, (center, c) in enumerate(zip(sequence.center, colors)):
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
            # handle = scatter_baseline(idx, sequence, idx, axis=axes[seed])
            handles.append(handle)

        axes[seed].set_ylabel("# Sequences")
    axes[seed].legend(handles=handles, labels=sequence.center)
    PIC.save_figure("seq_across_baselines", fig, config.sub_dir)


def scatter_baseline_patch(x, sequence, center_idx:int, distance:float=1., average_only:bool=False, **kwargs):
    if not average_only:
        scatter(x, sequence.baseline[center_idx], markerfacecolor="white", markersize=MS / 2, **kwargs)
        scatter(x+distance, sequence.patch[center_idx], markerfacecolor="white", markersize=MS / 2, **kwargs)

    return scatter([x, x+distance], [sequence.baseline_avg[center_idx], sequence.patch_avg[center_idx]], markersize=MS, **kwargs)
    scatter(x, sequence.baseline_avg[center_idx], markersize=MS, **kwargs)
    return scatter(x+distance, sequence.patch_avg[center_idx], markersize=MS, **kwargs)


def scatter_baseline(x, sequence, center_idx:int, distance:float=1., **kwargs):
    scatter(x, sequence.baseline[center_idx], markerfacecolor="white", markersize=MS / 2, **kwargs)
    return scatter(x, sequence.baseline_avg[center_idx], markersize=MS, **kwargs)


def scatter(x, data, axis:object=None, **kwargs):
    ax = axis if axis is not None else plt
    plot_to_scatter = {"ls": "None", "marker": "o"}
    plot_to_scatter.update(kwargs)
    try:
        line, = ax.plot(np.full(shape=len(data), fill_value=x), data, **plot_to_scatter)
    except TypeError:
        line, = ax.plot(x, data, **plot_to_scatter)
    return line


def bold_spines(ax, width:float=1):
    tick_params = {"width": width, "length": width * 3, "labelleft": True, "labelbottom": True}
    ax.tick_params(**tick_params)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('bottom', 'left'):
            ax.spines[s].set_linewidth(width)


if __name__ == "__main__":
    main()
