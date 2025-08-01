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
__version__ = '0.1c'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


import lib.pickler as PIC
import lib.universal as UNI

from plot.lib.frame import create_image
from plot.lib import plot_patch
from plot.lib import add_colorbar
from plot.constants import COLOR_MAP_ACTIVITY

from params import config


marker = ["o", "*", "^", "v", "s"]
bs_color = "magenta"
rcParams['lines.markersize'] = 5

degree_cmap = plt.cm.jet
min_degree = 575
max_degree = 850

save = True
#===============================================================================
# MAIN
#===============================================================================

def main():

    # =============================================================================
    # Density across sequence count and duration
    # Experimental: Idea is to have the same plot with duration over sequence count.
    # But as density and not as mean+-std
    # =============================================================================
    if UNI.yes_no("Plot densitiy of sequence count and duration?", False):
        percentage = config.PERCENTAGES[0]
        xmargin = 10
        ymargin = 10

        assert len(config.PERCENTAGES) == 2

        fig, axes = plt.subplots(
            ncols=len(config.PERCENTAGES) + 1,
            num="density",
            # figsize=(8, 3.5),
            tight_layout=False,
            sharey=True,
        )
        fig.suptitle("Activation densitiy across conditions")

        for ax in axes:
            if ax == axes[0]:
                ax.set_ylabel("Avg. duration [ms]")
                ax.set_yticks([220, 290, 360])
                ax.set_ylim([221, 365])

            ax.set_xlabel("Sequence count")
            ax.set_xticks([75, 90, 105])
            ax.set_xlim([66, 122])

        # Get the data - across radii and locations, but only for 1 percentage.
        tags = config.get_all_tags(radius=config.radius[0], weight_change=percentage)
        durations_across_tags = np.zeros((len(config.radius), len(tags)))
        sequence_count_across_tags = np.zeros((len(config.radius), len(tags)))

        for ip, p in enumerate(config.radius):
            tags = config.get_all_tags(radius=p, weight_change=percentage)

            for it, tag in enumerate(tags):
                durations, _sequence_count = get_durations_and_sequencecount(tag, config)
                durations_across_tags[ip, it] = durations.mean()
                sequence_count_across_tags[ip, it] = _sequence_count



        from scipy import stats
        # Find/set limits
        xmin, xmax = sequence_count_across_tags.min() - xmargin, sequence_count_across_tags.max() + xmargin
        ymin, ymax = durations_across_tags.min() - ymargin, durations_across_tags.max() + ymargin
        # Form grid
        X, Y = np.mgrid[xmin:xmax, int(ymin):int(ymax)]
        positions = np.vstack([X.ravel(), Y.ravel()])

        Zs = np.zeros((len(config.radius), *X.shape))
        for ip, p in enumerate(config.radius):
            values = np.vstack([sequence_count_across_tags[ip], durations_across_tags[ip]])
            kernel = stats.gaussian_kde(values, bw_method=0.75)
            Zs[ip] = np.reshape(kernel(positions).T, X.shape)

        # plotting
        imshow_kwargs = {
            "extent": [xmin, xmax, ymin, ymax],
            "origin": "lower",
        }
        extent=[xmin, xmax, ymin, ymax]
        for ip, p in enumerate(config.radius):
            axes[ip].imshow(Zs[ip].T, cmap=plt.cm.hot, **imshow_kwargs)

            axes[ip].scatter(sequence_count_across_tags[ip], durations_across_tags[ip])

        # density_difference = Zs[0].T - Zs[1].T
        # vmax = np.max(np.abs(density_difference))
        # im = axes[-1].imshow(density_difference, cmap=plt.cm.seismic, vmax=vmax, vmin=-vmax, **imshow_kwargs)
        # plt.colorbar(im)

    plt.show()


#===============================================================================
# SEQUENCE LANDSCAPE & Sequence difference
#===============================================================================

def plot_sequence_landscape(tag, config:object, plot_diff:bool=False) -> None:
    tags = None
    if isinstance(tag, (list, tuple)):
        tags = tag
        name = UNI.name_from_tag(tags[0])
        radius = UNI.radius_from_tag(tags[0])
    else:
        name = UNI.name_from_tag(tag)
        radius = UNI.radius_from_tag(tag)
    num = f"sequences_{name}_{radius}" if not plot_diff else f"sequences_diff_{name}_{radius}"
    if plt.fignum_exists(num):
        return
    # fig, (ax_seq, ax_grad) = plt.subplots(ncols=2, num=num, figsize=(12, 3.5))
    fig_seq, ax_seq = plt.subplots(num=f"{num}_seq")
    fig_seq.num = f"{num}_seq"
    fig_grad, ax_grad = plt.subplots(num=f"{num}_grad")
    fig_grad.num = f"{num}_grad"
    # ax_seq.set_title("Sequence count")
    ax_seq.set_xlabel("x")
    ax_seq.set_ylabel("y")
    ax_seq.set_xticks([10, 40, 70])
    ax_seq.set_yticks([10, 40, 70])
    ax_grad.set_xlabel("x")
    ax_grad.set_ylabel("y")
    ax_grad.set_xticks([10, 40, 70])
    ax_grad.set_yticks([10, 40, 70])
    if not tags:
        seq_count = _get_sequence_landscape(tag, config)
    else:
        seq_counts = []
        for t in tags:
            seq_count_tmp = _get_sequence_landscape(t, config)
            print(seq_count_tmp.shape)
            seq_counts.append(seq_count_tmp)
        seq_count = np.asarray(seq_counts, dtype=int).mean(axis=0)
        print(seq_count.shape)

    # if config.get_baseline_tag_from_tag(tag) == tag:
    if not plot_diff:
        cmap = COLOR_MAP_ACTIVITY
        cmap = "Blues"
        cmap = "hot_r"
        norm = (0, np.max(seq_count))
        im = create_image(seq_count.T, norm=(0, np.max(seq_count)), axis=ax_seq, cmap=cmap)
        cbar = add_colorbar(ax_seq, norm, cmap=cmap, ticks=[0, 5, 10, 15])
        cbar.set_label("Sequence count", rotation=270, labelpad=15)
        # cbar = plt.colorbar(im)
    else:
        seq_count_bs = _get_sequence_landscape(config.get_baseline_tag_from_tag(tag), config)
        if plot_diff:
            seq_diff = seq_count - seq_count_bs
            _max = np.max(np.abs(seq_diff))
            im = create_image(seq_diff.T, norm=(-_max, _max), cmap="seismic", axis=ax_seq)
        else:
            im = create_image(seq_count.T, norm=(0, seq_count.max()), axis=ax_seq)
        cbar = plt.colorbar(im)
    if name in config.center_range.keys():
        plot_patch(config.center_range[name], float(radius), config.rows)
    # cbar.ax.locator_params(nbins=5)

    # fig, ax = plt.subplots(num=f"gradient_{tag}")
    # ax_grad.set_title("Gradient of the sequence count")
    grad_seq_x, grad_seq_y = np.gradient(seq_count, edge_order=2)
    grad = np.stack((grad_seq_x, grad_seq_y))
    cmap = "hot_r"
    cmap = "YlOrRd"
    cmap = truncate_colormap(cmap, maxval=0.8)
    cmap = "seismic"
    cmap = truncate_colormap(cmap, minval=0.5, maxval=0.9)
    norm = (0, np.quantile(grad, q=0.95))
    grad_norm = np.linalg.norm(grad, axis=0)
    norm = (0, np.max(grad_norm))
    im = create_image(grad_norm.T, cmap=cmap, axis=ax_grad)
    cbar = add_colorbar(ax_grad, norm, cmap, ticks=[0, 5, 10, 15])
    cbar.set_label("Sequence count gradient", rotation=270, labelpad=15)
    # # cbar = plt.colorbar(im)
    # cbar.ax.locator_params(nbins=5)
    # plt.quiver(grad_seq_x.T, grad_seq_y.T, angles="xy", pivot="mid", scale_units="xy", scale=3)

    if save:
        # PIC.save_figure(num, fig, sub_directory=config.sub_dir)
        print(fig_seq.num)
        PIC.save_figure(fig_seq.num, fig_seq, sub_directory=config.sub_dir, transparent=True)
        PIC.save_figure(fig_grad.num, fig_grad, sub_directory=config.sub_dir, transparent=True)


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
    from plot.lib.frame import create_image
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

from matplotlib.colors import LinearSegmentedColormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    if isinstance(cmap, str):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(cmap)
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
#===============================================================================
# COUNT AND DURATION VS INDEGREE
#===============================================================================
def plot_seq_duration_over_indegree(config:object, feature:str=None) -> None:
    feature = feature if feature is not None else "duration"

    figname = f"{feature} over indegree"
    if not plt.fignum_exists(figname):
        fig, axes = plt.subplots(
            ncols=len(config.PERCENTAGES),
            num=figname,
            # figsize=(8, 3.5),
            tight_layout=False,
            sharey=True,
        )
        # fig.suptitle("Activation across conditions")
        axes = UNI.make_iterable(axes)
        for ax in axes:
            if ax == axes[0]:
                if "duration" in feature.lower():
                    ax.set_ylabel("Difference in avg. duration [ms]")
                elif "sequence" in feature.lower():
                    ax.set_ylabel("Difference in Sequence Count")
                else:
                    ax.set_ylabel("No feature set.")
            # if "duration" in feature.lower():
            #     ax.set_ylim(-85, 85)
            #     ax.set_yticks(np.arange(-50, 50+1, 50, dtype=int))
            # elif "sequence" in feature.lower():
            #     ax.set_ylim(-21, 21)
            #     ax.set_yticks(np.arange(-20, 20+1, 10, dtype=int))

            ax.set_xlabel("Mean Patch In-degree")
            ax.axhline(c=bs_color, lw=2, label="baseline")
            ax.set_xticks([700, 800])
    else:
        raise LookupError


    for i, p in enumerate(config.PERCENTAGES):

        _, std = _plot_feature_vs_indegree(config, config.baseline_tags, feature=feature, is_baseline=True)
        axes[i].axhline(std, c=bs_color, lw=2, ls="--")
        axes[i].axhline(-std, c=bs_color, lw=2, ls="--")
        axes[i].axhspan(-std, std, color=bs_color, alpha=0.075)



        tags_by_seed = config.get_all_tags(seeds="all", weight_change=[p])
        plt.sca(axes[i])
        # plt.title(f"Syn. change: {int(100*p):+}%")


        plot_kwargs = {
            "marker": "o",
            "capsize": 4,
        }
        for s, tag_seeds in enumerate(tags_by_seed):
            if "loc" in tag_seeds[0]:
                plot_kwargs.pop("markerfacecolor", None)
            # else:
            #     plot_kwargs["markerfacecolor"] = "k"
            mean, std = _plot_feature_vs_indegree(config, tag_seeds, feature=feature, **plot_kwargs)

    if "duration" in feature.lower():
        axes[0].set_ylim(-85, 85)
        axes[0].set_yticks(np.arange(-50, 50+1, 50, dtype=int))
    elif "sequence" in feature.lower():
        axes[0].set_ylim(-21, 21)
        axes[0].set_yticks(np.arange(-20, 20+1, 10, dtype=int))
    axes[1].legend()
    if save:
        PIC.save_figure(f"{figname}_{config.radius[0]}", fig, sub_directory=config.sub_dir, transparent=True)


def _plot_feature_vs_indegree(config:object, tag_across_seed:list, feature:str=None, is_baseline:bool=False, **plot_kwargs) -> None:
    # Initiate zeros for as many seeds
    duration = np.zeros(len(tag_across_seed))
    sequence_count = np.zeros(len(tag_across_seed))
    for seed, tag in enumerate(tag_across_seed):
        # Get spikes and durations for baseline
        bs_tag = config.get_baseline_tag_from_tag(tag)
        bs_spikes, bs_labels = PIC.load_spike_train(bs_tag, config)
        bs_durations = _get_durations(bs_spikes[:, 0], bs_labels) # [:, 0] -> only takes the time points


        if not is_baseline:
            # Get spikes and durations for patch
            spikes, labels = PIC.load_spike_train(tag, config)
            durations = _get_durations(spikes[:, 0], labels) # [:, 0] -> only takes the time points

            duration[seed] = durations.mean() - bs_durations.mean()
            sequence_count[seed] = labels.max() - bs_labels.max()
        else:
            duration[seed] = bs_durations.mean()
            sequence_count[seed] = bs_labels.max()

    if "duration" in feature.lower():
        feature = duration
    else:
        feature = sequence_count

    mean = feature.mean()
    std = feature.std(ddof=1)
    if not is_baseline:
        indegree = get_indegree(config, tag_across_seed)
        print(25*"-")
        logger.info(f"Indegree: {tag_across_seed[0]} -->  {indegree}")
        color = map_indegree_to_color(indegree)
        plt.errorbar(indegree, mean, yerr=std / np.sqrt(feature.size), color=color, **plot_kwargs)
        # for m in feature:
        #     plt.scatter(indegree, m, color=color)
    if not is_baseline:
        del spikes, labels
    del bs_spikes, bs_labels
    return mean, std


def _get_durations(times:np.ndarray, labels:np.ndarray) -> np.ndarray:
    seq_count = labels.max()
    durations = np.zeros(seq_count)
    for l in range(seq_count):
        idx = labels == l
        durations[l] = times[idx].max() - times[idx].min()
    return durations


#===============================================================================
# COUNT AND DURATION
#===============================================================================

def plot_count_and_duration(config:object):
    figname = "duration over sequence count"
    if not plt.fignum_exists(figname):
        fig, axes = plt.subplots(
            ncols=len(config.PERCENTAGES),
            num=figname,
            # figsize=(8, 3.5),
            # tight_layout=True,
            sharey=True,
        )
        # fig.suptitle("Activation across conditions")
        axes = UNI.make_iterable(axes)
        for ax in axes:
            if ax == axes[0]:
                ax.set_ylabel("Avg. duration [ms]")
                ax.set_yticks([230, 290, 350])
                ax.set_ylim([217, 363])

            ax.set_xlabel("Sequence count")
            ax.set_xticks([85, 95, 105])
            ax.set_xlim([78, 111])
    else:
        raise LookupError

    for i, p in enumerate(config.PERCENTAGES):
        tags_by_seed = config.get_all_tags(seeds="all", weight_change=[p])
        plt.sca(axes[i])
        # plt.title(f"Syn. change: {int(100*p):+}%")



        plot_kwargs = {
            "marker": ".",
            "capsize": 2,
            # "markeredgecolor": "k",
            # "markerfacecolor": "k"
            }
        bs_kwargs = {"label": "baseline", "zorder": 20, "ls": "none"}
        _plot_count_vs_duration(config, tags_by_seed[0], is_baseline=True, **plot_kwargs, **bs_kwargs)

        # tags_by_seed = config.get_all_tags(weight_change=[p])
        # _plot_count_vs_duration(config, tags_by_seed, **plot_kwargs)
        for s, tag_seeds in enumerate(tags_by_seed):
            _plot_count_vs_duration(config, tag_seeds, **plot_kwargs)

    plt.legend(handletextpad=0.1)

    if save:
        PIC.save_figure(f"{figname}_{config.radius[0]}", fig, sub_directory=config.sub_dir, transparent=True)


def _plot_count_vs_duration(config:object, tag_across_seed:list, is_baseline:bool=False, **plot_kwargs) -> None:
    duration = np.zeros(len(tag_across_seed))
    sequence_count = np.zeros(len(tag_across_seed))
    for seed, tag in enumerate(tag_across_seed):
        if is_baseline:
            tag = config.get_baseline_tag_from_tag(tag)

        durations, _sequence_count = get_durations_and_sequencecount(tag, config)
        duration[seed] = durations.mean()
        sequence_count[seed] = _sequence_count

    if is_baseline:
        color = bs_color
    else:
        indegree = get_indegree(config, tag_across_seed)
        color = map_indegree_to_color(indegree)
    # plt.scatter(sequence_count, duration, color=color)
    return plt.errorbar(sequence_count.mean(), duration.mean(),
                  # xerr=sequence_count.std(), yerr=duration.std(),
                  xerr=sequence_count.std(ddof=1) / np.sqrt(sequence_count.size),
                  yerr=duration.std(ddof=1) / np.sqrt(duration.size),
                  color = color, **plot_kwargs)


def get_durations_and_sequencecount(tag:str, config:object) -> tuple:
    spikes, labels = PIC.load_spike_train(tag, config)
    durations = _get_durations(spikes[:, 0], labels)
    return durations, labels.max()

#===============================================================================
#
#===============================================================================


def get_indegree(config:object, tags:list):
    import lib.dopamine as DOP

    name = UNI.name_from_tag(tags[0])
    center = config.center_range[name]


    radius = UNI.radius_from_tag(tags[0])
    patch = DOP.circular_patch(config.rows, tuple(center), float(radius))
    patch = patch.reshape((config.rows, config.rows))


    if hasattr(get_indegree, "conn"):
        conn = get_indegree.conn
        logger.info("Re-use Conn. Matrix")
    else:
        logger.info("Load Conn. Matrix")
        from lib.connectivitymatrix import ConnectivityMatrix
        conn = ConnectivityMatrix(config)
        get_indegree.conn = conn

    if hasattr(get_indegree, "indegree"):
        indegree = get_indegree.indegree
        logger.info("Use cached indegree")
    else:
        _, indegree = conn.degree(conn.connections[:config.rows**2, :config.rows**2])
        get_indegree.indegree = indegree

    # MEAN
    patch_indegree = np.mean(indegree[patch]) * config.synapse.weight

    return patch_indegree


def map_indegree_to_color(indegree:float) -> float:
    indegree = min_degree if indegree < min_degree else indegree
    indegree = max_degree if indegree > max_degree else indegree
    color = degree_cmap((indegree - min_degree) / (max_degree - min_degree))
    return color


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
