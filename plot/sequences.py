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
import cflogger
logger = cflogger.getLogger()


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
    from params import PerlinConfig, GateConfig, SelectConfig, BrianConfig, GateRepeatConfig, RandomLocationConfig

    cf = GateRepeatConfig()

    if yes_no("Plot detected sequences?"):
        plot_db_sequences(cf, cf.get_all_tags())
    if yes_no("Plot sequences across baseline locations?"):
        plot_baseline_sequences(cf)
    if yes_no("Plot sequence correlations?"):
        for tags in cf.get_all_tags(seeds="all"):
            plot_sequence_correlations(cf, tags, add_detailed_plot=True)

    if yes_no("Plot sequence count and duration?"):
        plot_count_and_duration(cf)
    if yes_no("Plot difference across sequence counts?"):
        plot_seq_diff(cf)


########## DEVELOPMENT: ANALYZE CORRELATION (GATE/SELECT) #############################################################

def plot_detailed_correlations(pre:np.ndarray, post:np.ndarray, id_:str):

    from analysis import sequence_correlation as SC
    correlator = SC.SequenceCorrelator(None)
    fig, (ax1, ax2) = plt.subplots(2, num=id_)
    for s, center in zip((pre, post), id_.split("_")):
        # Plain Spike train convolved with the kernel
        train, t_axis, kernel = correlator._convolve_gauss_kernel(s)
        ax1.plot(t_axis, train, label=center)

    ax1.set_xlabel("Time [s]")
    ax1.set_title("Spike trains (Gauss kernel)")
    ax1.legend()

    correlation_normalized, time_axis_correlation = correlator._correlate_sequence_times(pre, post)
    ax2.plot(time_axis_correlation, correlation_normalized)
    ax2.set_xlim(-50, 50)
    ax2.set_title("Smoothed correlation")
    ax2.set_xlabel("Time shift")
    ax2.set_ylabel("Normalized correlation (no. of spikes)")
    fig.tight_layout()



def get_transmission_factor(correlation:np.ndarray, no_of_spikes):
    # Calculate transmission fraction: How many spikes of the pre spot occured in correlation at the post spot
    try:
        transmission_fraction = np.max(correlation) / no_of_spikes
    except ZeroDivisionError:
        transmission_fraction = 0
    return transmission_fraction


############################ TODO: Split calculation and the plotting ##################################

def correlation_from_sequence(sequence:object, len_center):
    lags = np.zeros((len_center, len_center, 2)) # Off diagonals are interesting
    transmission = np.zeros((len_center, len_center, 2)) # Off diagonals are interesting

    # Run across off-diagonals
    for pre, post in itertools.permutations(range(len_center), 2):
        for i, (corr, spike_times) in enumerate(((sequence.correlations_baseline, sequence.baseline_times),
                                                 (sequence.correlations_patch, sequence.patch_times))):
                time, correlation = corr[pre, post]

                max_time_shift = SequenceConfig.max_time_shift
                local_correlation_idx = np.bitwise_and(time <= max_time_shift, time >= -max_time_shift)

                max_corr_idx = correlation[local_correlation_idx].argmax()
                time_lag = time[local_correlation_idx][max_corr_idx]
                lags[pre, post, i] = time_lag

                transmission_fraction = get_transmission_factor(correlation, spike_times[pre].size)
                transmission[pre, post, i] = transmission_fraction
    return lags, transmission



def plot_sequence_correlations(config:object, tags:list, add_detailed_plot:bool=False):
    """
    Plot the time lag in the first column, and the transmission factor in the second column.
    tags: list of all seeds across one condition, e.g.: repeater_6_50_20_0
    """

    name, _  = UNI.split_seed_from_tag(tags[0])
    fig, (ax0, ax1) = plt.subplots(ncols=2, **SequenceConfig.correlation(name))

    labels = []
    # Tags just differ in the seed
    for idx, tag in enumerate(tags):
        sequence = PIC.load_db_cluster_sequence(tag, sub_directory=config.sub_dir)
        len_center = len(sequence.center)
        logger.info(f"{tag}: analyzing {len_center} center...")
        lags, transmission = correlation_from_sequence(sequence, len_center)
        # lags/transmission have shape (pre, post, idx) with idx being 0 for baseline, 1 for patch

        for pre, post in itertools.permutations(range(len_center), 2):

            plot_kwargs = _get_markup(pre, post)

            for i, (corr, spike_times) in enumerate(zip((sequence.correlations_baseline, sequence.correlations_patch),
                                                        (sequence.baseline_times, sequence.patch_times))):

                # Labels
                label = None if i > 0 else f"{sequence.center[pre]} - {sequence.center[post]}"
                label_tmp = label if label not in labels else None
                labels.append(label)

                x_shift = i + idx * SequenceConfig.increment_between_seeds
                ax0.plot(x_shift, lags[pre, post][i], **plot_kwargs, label=label_tmp)
                ax1.plot(x_shift, transmission[pre, post][i], **plot_kwargs)

            if add_detailed_plot:
                if idx == 0:
                    plot_detailed_correlations(sequence.baseline_times[pre], sequence.baseline_times[post], f"{sequence.center[pre]}_{sequence.center[post]}")

        ax0.legend()
        ax0.set_ylim(-SequenceConfig.max_time_shift, SequenceConfig.max_time_shift)
        ax0.set_title("Around 0: Baseline, Aroud 1: Patched, Jitter is across seeds")

        ax1.set_title("Transmission fraction: # how many of pre to post.")
        ax1.set_ylim(0, 1)
        PIC.save_figure(f"correlations_of_{name}", fig, config.sub_dir)



def _get_markup(pre:int, post:int) -> dict:
    # Fix the color for any combination of pre and post
    markup = {}
    markup["color"] = color_cycle[pre + post - 1]
    markup["marker"] = SequenceConfig.marker.default if pre < post else SequenceConfig.marker.alternative
    return markup




########## Sequences by different methods ##############################################################################


def plot_db_sequences(config, tags:list):
    """Plots the number of detected sequences using different methods (thresholding, mean thresholding, clustering)."""
    tags = UNI.make_iterable(tags)
    for tag in tags:
        fig, ax = plt.subplots(num=tag, figsize=(3, 3))

        try:
            # plot_sequences(config, tag, load_method=PIC.load_db_sequence, axis=ax)
            plot_sequences(config, tag, load_method=PIC.load_db_sequence, axis=ax, average_only=True, ls="-")
        except FileNotFoundError:
            logger.error("Cannot plot detected sequences in patch simulation, as the file is missing.")

        try:
            plot_sequences(config, tag, load_method=PIC.load_db_cluster_sequence, axis=ax, marker="*", average_only=True, ls="--")
        except FileNotFoundError:
            logger.error("Cannot plot detected sequences by cluster, as the file is missing.")
        PIC.save_figure(f"seq_db_{tag}", fig, config.sub_dir)


def plot_sequences(config:object, tag:str, axis, load_method=None, sequence=None, **plot_kwargs):
    sequence = load_method(tag, sub_directory=config.sub_dir) if sequence is None else sequence
    handles = []
    for idx, (center, c) in enumerate(zip(sequence.center, COLORS)):
        handle = scatter_baseline_patch(0., sequence, idx, distance=.4, c=c, axis=axis, **plot_kwargs)
        handles.append(handle)
    axis.set_ylabel("# sequences")
    axis.set_xticks([.0, .4], labels=["w/o patch", "w/ patch"])
    axis.set_xlim([-.05, .45])
    axis.set_ylim(bottom=-1)
    #axis.legend(handles=handles, labels=sequence.center)
    plt.tight_layout()


def plot_seq_diff(config:object):
    from plot.activity import create_image
    tags = config.get_all_tags()

    for tag in tags:
        sequence = PIC.load_db_cluster_sequence(tag, sub_directory=config.sub_dir)

        coordinates_bs = sequence.baseline_spikes[:, 1:]
        H_bs, _, _ = np.histogram2d(*coordinates_bs.T, bins=np.arange(-0.5, config.rows))

        coordinates = sequence.patch_spikes[:, 1:]
        H, _, _ = np.histogram2d(*coordinates.T, bins=np.arange(-0.5, config.rows))

        plt.figure(tag)
        plt.title(f"{tag} with {sequence.patch_labels.size} and {sequence.baseline_labels.size}")
        im = create_image(H.T - H_bs.T, norm=(-200, 200), cmap="seismic")
        plt.colorbar(im)
        break



def plot_count_and_duration(config:object):
    fig, (ax_count, ax_duration, ax_index) = plt.subplots(ncols=3)

    marker = ["o", "*", "^", "v"]

    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = len(config.get_all_tags(seeds="all"))
    colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    for s, tag_seeds in enumerate(config.get_all_tags(seeds="all")):
        for tag in tag_seeds:
            _, seed = UNI.split_seed_from_tag(tag)
            seed = int(seed)
            sequence = PIC.load_db_cluster_sequence(tag, sub_directory=config.sub_dir)
            seq_count = sequence.patch_labels.max()
            ax_count.plot(s, seq_count, marker=marker[seed], label=tag, color=colors[s])
            seq_count_bs = sequence.baseline_labels.max()
            ax_count.plot(-1, seq_count_bs, marker=marker[seed], color="k")

            times = sequence.patch_spikes[:, 0]

            durations = np.zeros(seq_count)
            for l in range(seq_count):
                idx = sequence.patch_labels == l
                durations[l] = times[idx].max() - times[idx].min()
            ax_duration.plot(s, durations.mean(), marker=marker[seed], label=tag, color=colors[s])

            times = sequence.baseline_spikes[:, 0]
            durations_bs = np.zeros(seq_count_bs)
            for l in range(seq_count_bs):
                idx = sequence.baseline_labels == l
                durations_bs[l] = times[idx].max() - times[idx].min()
            ax_duration.plot(-1, durations_bs.mean(), marker=marker[seed], color="k")

            ax_index.plot(s, seq_count * durations.mean(), marker=marker[seed], label=tag, color=colors[s])
            ax_index.plot(-1, seq_count_bs * durations_bs.mean(), marker=marker[seed], color="k")
    # ax_count.legend()
    # ax_duration.legend()
    # ax_index.legend()




########################################################################################################################



def plot_baseline_sequences(config:object)->None:
    """
    Loads the # of sequences detected by DBScan.
    """
    fig, axes = plt.subplots(ncols=len(config.simulation_seeds), num="baseline_sequences", figsize=SequenceConfig.figsize_baseline)
    fig.suptitle("# Sequences across baselines with different center (DBScan)")
    for seed in config.simulation_seeds:
        tag = config.baseline_tag(seed)
        try:
            sequence = PIC.load_db_sequence(tag, sub_directory=config.sub_dir)
        except FileNotFoundError:
            logger.error("Cannot plot detected sequences in baseline simulation, as the file is missing.")
            continue

        handles = []
        for idx, (center, color) in enumerate(zip(sequence.center, color_cycle)):
            handle = scatter_baseline(idx, sequence, idx, c=color, axis=axes[seed])
            # handle = scatter_baseline(idx, sequence, idx, axis=axes[seed])
            handles.append(handle)

        axes[seed].set_ylabel("# Sequences")
    axes[seed].legend(handles=handles, labels=sequence.center)
    PIC.save_figure("seq_across_baselines", fig, config.sub_dir)


def scatter_baseline_patch(x, sequence, center_idx:int, distance:float=1., average_only:bool=False, **kwargs):
    """Scatters the individual points and the mean of the baseline and the patch condition."""
    plot_to_scatter = {"markerfacecolor": "white", "markersize": MS / 2} # Default values
    plot_to_scatter.update(kwargs)
    if not average_only:
        scatter(x, sequence.baseline[center_idx], **plot_to_scatter)
        scatter(x+distance, sequence.patch[center_idx], **plot_to_scatter)

    return scatter([x, x+distance], [sequence.baseline_avg[center_idx], sequence.patch_avg[center_idx]], markersize=MS, **kwargs)

def scatter_baseline(x, sequence, center_idx:int, distance:float=1., **kwargs):
    """Scatters the individual points and the mean."""
    plot_to_scatter = {"markerfacecolor": "white", "markersize": MS / 2}
    plot_to_scatter.update(kwargs)
    scatter(x, sequence.baseline[center_idx], **plot_to_scatter)
    return scatter(x, sequence.baseline_avg[center_idx], markersize=MS, **kwargs)


def scatter(x:np.ndarray, data:np.ndarray, axis:object=None, **kwargs):
    # TODO: What is the benefit (except having a defaults?)
    ax = axis if axis is not None else plt
    plot_to_scatter = {"ls": "None", "marker": "o"}
    plot_to_scatter.update(kwargs)
    try:
        line, = ax.plot(np.full(shape=len(data), fill_value=x), data, **plot_to_scatter)
    except TypeError:
        line, = ax.plot(x, data, **plot_to_scatter)
    return line


def yes_no(question:str):
    answer = input(question + " (y/n)")
    return answer.lower().strip() == "y"

if __name__ == "__main__":
    main()
    plt.show()
