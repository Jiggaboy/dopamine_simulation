#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import copy
from itertools import pairwise
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


from lib.universal import dotdict
from plot.lib import remove_spines_and_ticks, remove_topright_spines, add_topright_spines
from lib.connectivitymatrix import ConnectivityMatrix
from params import config
import lib.pickler as PIC
from plot.lib.frame import create_image
from plot.lib.basic import add_colorbar, plot_patch_from_tag, add_colorbar_from_im
from plot.constants import COLOR_MAP_ACTIVITY, NORM_ACTIVITY, COLOR_MAP_DIFFERENCE

from figure_generator.in_out_degree import calculate_direction, plot_shift_arrows
from plot.constants import *

#===============================================================================
# CONSTANTS
#===============================================================================
figsize = (17.6*cm, 15*cm)


# PANEL NETWORK LAYOUT
side_length = 6

# PANEL CONNECTIVITY DISTRIBUTION
nrows = 40
MAX_HIST = 8
BIN_WIDTH = 1


N_rate = 15
RATE_THRESHOLD = .5
N_syn = 23
SYN_THRESHOLD = 50 # The difference between H0 and ext. drive

xyticks = (10, 50, 90)

indegree_low  = 725  # Some connections are as low as 650
indegree_high = 1465 # Some degree are as high as 1450  

indegree_ticks = (800, 1000, 1200, 1400)

t_low  = 1000
t_high = 1750
    
filename = "figure1"
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    # fig = plt.figure(figsize=figsize)
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 1])

    gs_top = gs[0].subgridspec(nrows=2, ncols=5, height_ratios=[1, 1.2], width_ratios=[1, .3, 1, .25, 1])
    gs_bottom = gs[1].subgridspec(nrows=1, ncols=4, width_ratios=[1, .4, 1, 1])

    ### NETWORK LAYOUT
    ax = fig.add_subplot(gs_top[0, 0])
    ax.set_title("Network Layout")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right")
    
    ax.set_xlim(-0.5, side_length-0.5)
    ax.set_xticks(np.arange(0, side_length), ["...", *np.arange(20, 20 + side_length-2), "..."])
    ax.set_ylim(-0.5, side_length-0.5)
    ax.set_yticks(np.arange(0, side_length), ["...", *np.arange(20, 20 + side_length-2), "..."])
    panel_network_layout(ax, side_length)
    ax.legend()
    # return
    ### CONNECTIVITY DISTRIBUTION
    ax = fig.add_subplot(gs_top[0, 2])
    ax.set(title="Connectivity Profile", xlabel="Static (symmetric)", ylabel="Shifted (asymmetric)",
           xlim=(0, nrows), ylim=(0, nrows))
    panel_connectivity_distribution(ax)
    
    ### SIMPLEX NOISE
    ax = fig.add_subplot(gs_top[0, 4])
    ax.set(title="Simplex Noise", xlabel="X", ylabel="Y")
    panel_simplex_noise(ax, config)
    
    ### INDEGREE 
    ax = fig.add_subplot(gs_top[1, 0])
    add_topright_spines(ax)
    ax.set(title="In-Degree", xlabel="X", ylabel="Y", xticks=xyticks, yticks=xyticks)
    im = panel_indegree(ax, config)
    cbar = add_colorbar_from_im(ax, im)
    cbar.set_ticks(indegree_ticks)
    cbar.set_label("In-degree", rotation=270, labelpad=8)
    
    ### AVG ACTIVITY
    ax = fig.add_subplot(gs_top[1, 2])
    add_topright_spines(ax)
    ax.set(title="Avg. Activity", xlabel="X", ylabel="Y",
           xticks=xyticks, yticks=xyticks)
    im = panel_avg_activity(ax, config)
    
    cbar = add_colorbar_from_im(ax, im)
    cbar.set_label("Avg. activity", rotation=270, labelpad=8)
    cbar.set_ticks([0.0, 0.2, 0.4])
    

    
    ### INDEGREE HIST
    ax = fig.add_subplot(gs_top[1, 4])
    remove_topright_spines(ax)
    ax.set(title="Indegree Distribution", xlabel="In-degree",
           xticks=(750, 1000, 1250), yticks=(0, 500, 1000, 1500), ylim=(0, 2000), xlim=(indegree_low, indegree_high))
    ax.set_ylabel("Occurrence")#, labelpad=4)
    panel_hist_indegree(ax, config)
    
    ### RATE HIST
    ax_rate = fig.add_subplot(gs_bottom[:2])
    ax_hist = fig.add_subplot(gs_bottom[2])
    remove_topright_spines(ax_rate)
    remove_topright_spines(ax_hist)
    
    ax_rate.set(title="Rate over Time", xlabel="Time [ms]", ylabel="Rate", 
                xlim=(t_low, t_high), ylim=(0, 1))
    ax_hist.set(title="Rate Distribution", xlabel="Density", ylim=(0, 1))
    ax_hist.tick_params(labelleft=False)
    
    panel_hist_rate(ax_rate, ax_hist, config)
    
    ### INPUT HIST
    ax = fig.add_subplot(gs_bottom[-1])
    remove_topright_spines(ax)
    ax.set(title="Synaptic Input", xlabel="Synaptic input", ylabel="Density", 
                ylim=(1e-8, 1e2))
    panel_hist_input(ax, config)
    
    
    PIC.save_figure(filename, fig, transparent=True)

#===============================================================================
# METHODS
#===============================================================================
def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")


def panel_network_layout(ax:object, side_length:int):
    pos = np.arange(side_length)
    x_pos_exc, y_pos_exc = np.meshgrid(pos, pos)


    iside = np.arange(0.5, side_length, 2)
    x_pos_inh, y_pos_inh = np.meshgrid(iside, iside)

    ax.scatter(x_pos_exc, y_pos_exc, color=KTH_PINK, marker="o", label="exc. neuron")
    ax.scatter(x_pos_inh, y_pos_inh, color=KTH_BLUE, marker="x", label="inh. neuron")


def panel_connectivity_distribution(ax:object):
    # Network
    center = np.asarray((nrows // 2, nrows // 2), dtype=int)
    stdIE_ratio = 6 / 2.75

    ## Targets
    targets = dotdict({
        "std": 2.75,
        "n_conn": 200,
    })
    target_style = {
        "marker": "o",
        "ms": 2.4,
        "linestyle": "None",
        "mec": "k",
        "mew": 0.2,
    }

    SHIFT = 2

    # Targets (style)
    C_TARGET = KTH_YELLOW
    C_TARGET_SHIFTED = KTH_GREEN

    neuron_style = {
        "ms": 6,
        "color": KTH_PINK,
        "marker": "o",
        "ls": "None",
        "mec": "k",
        "zorder": 10,
    }


    #### STYLE HISTOGRAMS
    C_INH_HIST = KTH_BLUE
    C_FULL_HIST = KTH_GREY


    #### SCALEBAR
    X_SCALEBAR = 8
    Y_SCALEBAR = 35
    WIDTH_SCALEBAR = 2

    scalebar_style = {"color": "black", "linewidth": 2}
    remove_spines_and_ticks(ax)


    logger.info("Scatter shifted and unshifted targets.")
    scatter_targets(ax, center=center, shift=0, color=C_TARGET, **targets, **target_style)
    scatter_targets(ax, center=center, shift=[SHIFT, -SHIFT], color=C_TARGET_SHIFTED, **targets, **target_style)
    neuron = ax.plot(*center, **neuron_style)

    logger.info("Histogram of the static targets.")
    hist_params = dict(**targets, **{
        "axis": "x",
        })
    hist_params["n_conn"] *= 250

    logger.info("Hist. of exc. distribution")
    exc_dist, bins = get_hist_of_normal(center, std=targets["std"], size=hist_params["n_conn"])
    exc_handle_unshifted = hist_dist(ax, bins, exc_dist, color=C_TARGET, axis="x")

    # exc_dist, bins, exc_handle_unshifted = hist_exc_dist(ax, shift=0., center=center, color=C_TARGET, **hist_params)
    inh_dist, bins, inh_handle = hist_inh_dist(ax, center=center, color=C_INH_HIST, factor=stdIE_ratio, **hist_params)
    joint_dist = hist_dist(ax, bins, exc_dist - inh_dist, axis=hist_params["axis"], color=C_FULL_HIST) 

    
    logger.info("Histogram of the shifted targets.")
    hist_params["axis"] = "y"

    exc_dist_shifted, bins, exc_handle_shifted = hist_exc_dist(ax, shift=-SHIFT, center=center, color=C_TARGET_SHIFTED, **hist_params)
    inh_dist, bins, inh_handle = hist_inh_dist(ax, center=center, color=C_INH_HIST, factor=stdIE_ratio, **hist_params)
    joint_dist = hist_dist(ax, bins, exc_dist_shifted - inh_dist, axis=hist_params["axis"], color=C_FULL_HIST) 

    ax.legend([*exc_handle_shifted, *exc_handle_unshifted, *inh_handle, *neuron],
              ["exc. (shifted)", "exc. (static)", "inh.", "presyn. neuron"],
               loc="upper right",
              )

    ax.axhline(4, xmin=0.1, ls="--", c="k", zorder=-5)
    ax.axvline(4, ymin=0.1, ls="--", c="k", zorder=-5)
    ax.axhline(nrows // 2 - SHIFT, xmin=0.1, xmax=0.575, ls="--", c="k", zorder=5)
    ax.axvline(nrows // 2, ymin=0.1, ymax=0.575, ls="--", c="k", zorder=5)
    # ax.axhline(nrows // 2 - SHIFT, xmin=0.1, xmax=0.575, ls="--", c=KTH_GREEN, zorder=5)
    # ax.axvline(nrows // 2, ymin=0.1, ymax=0.575, ls="--", c=KTH_YELLOW, zorder=5)


def panel_simplex_noise(ax:object, config:object):
    tmp_config = copy.copy(config)
    tmp_config.landscape = copy.copy(config.landscape)
    tmp_config.landscape.params = dict(config.landscape.params)

    tmp_config.rows = 26
    tmp_config.landscape.params["size"] = 2
    tmp_config.landscape.params["base"] = 8

    conn = ConnectivityMatrix(tmp_config)

    bins = 6
    ax.set_xlim(-0.5, bins-0.5)
    ax.set_xticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
    ax.set_ylim(-0.5, bins-0.5)
    ax.set_yticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
    plot_shift_arrows(conn.shift, ax=ax)


def panel_indegree(ax:object, config:object):
    conn = ConnectivityMatrix(config)
    indegree, _ = conn.degree(conn._EE)
    indegree = indegree * config.synapse.weight

    return ax.imshow(indegree,
                    origin="lower",
                    cmap=CMAP_DEGREE,
    )


def panel_avg_activity(ax:object, config:object):
    tags = config.baseline_tags

    # Gather all rates
    rates = []
    for tag in tags:
        logger.info(f"Load {tag}...")
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)
        rates.append(avgRate)
    rates = np.asarray(rates)

    # Average if more than one run
    if rates.ndim > 1:
        rates = rates.mean(axis=0)

    norm = (0, 0.5)
    cmap = COLOR_MAP_ACTIVITY

    return create_image(rates, norm, cmap, axis=ax)
    
    
def panel_hist_indegree(ax:object, config:object, seed_range:int=2, bins:int=15) -> None:
    ytext = 1750
     
    original_seed = config.landscape.seed
    hists = np.zeros((seed_range, bins))
    conns = np.zeros((seed_range, config.no_exc_neurons))
    for s in range(seed_range):
        config.landscape.seed = s
        conn = ConnectivityMatrix(config)
        indegree, _ = conn.degree(conn._EE) 
        indegree *= config.synapse.weight
        # Clip the array to improve the visuals of the plot
        indegree = np.clip(indegree, indegree_low, indegree_high)
        conns[s] = np.sort(indegree.flatten())
        hists[s], edges = np.histogram(indegree, bins=bins, range=(indegree_low, indegree_high))        
    ax.bar(edges[:-1], hists.mean(axis=0), #yerr=hists.std(axis=0), 
           align="edge", width=0.8*(edges[1]-edges[0]), 
           #capsize=4, error_kw={"elinewidth": 2}
        )
    
    ax.axvline(conns.mean(), ls="dotted", c="k", label="mean", ymax=0.8)
    
    # splits = np.linspace(20, 80, 4)
    # percentiles = np.zeros((seed_range, splits.size))
    # for c, conn in enumerate(conns):
    #     for s, split in enumerate(splits):
    #         percentiles[c, s] = np.percentile(conn, split)
    #
    # from figure_generator.figure2 import map_indegree_to_color
    # for sep in percentiles.mean(axis=0):
    #     color = map_indegree_to_color(sep)
    #     ax.axvline(sep, ymax=0.8, ls="--", c=color)
    
    
    
    lows = np.zeros(seed_range)
    highs = np.zeros(seed_range)
    percentile_1 = int(config.no_exc_neurons * 0.01)
    for c, conn in enumerate(conns):
        lows[c] = conn.min()#conn[:percentile_1].mean()
        highs[c] = conn.max()#conn[-percentile_1:].mean()
    
    separator = np.linspace(lows.mean(), highs.mean(), 5+1)
    from figure_generator.figure2 import map_indegree_to_color
    for sep in separator[1:-1]:
        color = map_indegree_to_color(sep, indegree_low, indegree_high)
        ax.axvline(sep, ymax=1, ls="--", c=color)
        
    text_kwargs = {"zorder": 12, "verticalalignment": "center", "backgroundcolor": "None", "fontsize":8}
    names = "low", "low-\nmid", "mid", "mid-\nhigh", "high"
    ax.text(795, ytext, names[0], horizontalalignment="center", **text_kwargs)
    ax.text(separator.mean(), ytext, names[2], horizontalalignment="center", **text_kwargs)
    ax.text(1390, ytext, names[-1], horizontalalignment="center", **text_kwargs)
    # ax.text(separator[2:4].mean(), ytext, names[2], horizontalalignment="center", **text_kwargs)
    # for (l, r), name in zip(pairwise(separator[1:-1]), names[1:-1]):
    #     center = (l + r) / 2
    #     ax.text(center, ytext, name, horizontalalignment="center", **text_kwargs)
    ax.legend(
        loc="center right",
        # handlelength = 1.,
        # borderpad = 0.2,
        # labelspacing = 0.1,
    )
    config.landscape.seed = original_seed


def panel_hist_rate(ax_rate:object, ax_hists:object, config:object) -> None:
    
    # ax_hist_low, ax_hist_high = ax_hists
    # ax_hist_low.set_title("Rates")
    # ax_hist_low.set_xlabel("probability?")
    # ax_hist_low.set_xlim(0, 1)
    # ax_hist_low.set_xticks((0, 1))
    # ax_hist_high.set_xlim(13, 14)
    # ax_hist_high.set_xticks((13.5, ))
    # # Hide spines between axes
    # ax_hist_low.spines["right"].set_visible(False)
    # ax_hist_low.spines["top"].set_visible(False)
    # ax_hist_high.spines["left"].set_visible(False)
    # ax_hist_high.spines["right"].set_visible(False)
    # ax_hist_high.spines["top"].set_visible(False)
    # ax_hist_low.tick_params(labelright=False)
    # ax_hist_low.tick_params(labelleft=False)
    # ax_hist_high.tick_params(labelleft=False)
    # ax_hist_high.tick_params(
    #     axis='y',          
    #     which='both',    
    #     left=False,      
    #     right=False,         
    # )
    # add_break_marks(ax_hists)

    bins = np.linspace(0, 1, N_rate+1, endpoint=True) 
    
    bs_rates = np.zeros((len(config.baseline_tags), N_rate))
    portions = np.zeros(len(config.baseline_tags))
    
    for t, tag in enumerate(config.baseline_tags):
        rate = PIC.load_rate(tag, skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)
        H, edges = np.histogram(rate.ravel(), density=True, bins=bins)
        bs_rates[t] = H
        
        portion = H[edges[:-1] >= RATE_THRESHOLD].sum() / H.sum()
        portions[t] = portion
        
    handles = []
    for i in [96, 1727, 3661, 6941, 8342]:
        handle = ax_rate.plot(np.arange(t_low, t_high), rate[i][t_low:t_high], label=f"#{i:4}")
        handles.append(handle[0])
    ax_rate.legend(handles=handles, prop={'family': 'monospace'}) #handletextpad=0.6, 
    
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    
    # ax_hist_low.barh(bin_centers, bs_rates.mean(axis=0), xerr=bs_rates.std(axis=0), height=0.05)
    # ax_hist_high.barh(bin_centers, bs_rates.mean(axis=0), xerr=bs_rates.std(axis=0), height=0.05)
    #
    # ax_hist_low.axhline(config.analysis.sequence.spike_threshold, ls="--", c="k")
    ax_hists.barh(bin_centers, bs_rates.mean(axis=0), xerr=bs_rates.std(axis=0, ddof=1), height=0.05, log=True)
    # ax_hist_high.barh(bin_centers, bs_rates.mean(axis=0), xerr=bs_rates.std(axis=0), height=0.05)
    
    ax_hists.axhline(config.analysis.sequence.spike_threshold, ls="--", c="k")
    ax_hists.text(1, config.analysis.sequence.spike_threshold, "threshold", style="italic",
                  verticalalignment="bottom", horizontalalignment="center")



def panel_hist_input(ax:object, config:object) -> None:
    bins = np.linspace(-225, 325, N_syn+1, endpoint=True)
    
    syn_inputs = np.zeros((len(config.baseline_tags), (config.no_inh_neurons+config.no_exc_neurons)*config.sim_time))
    hist_inputs = np.zeros((len(config.baseline_tags), N_syn))
    for t, tag in enumerate(config.baseline_tags):
        synaptic_input = PIC.load_synaptic_input(tag, sub_directory=config.sub_dir)
        syn_inputs[t] = synaptic_input.ravel()
        H, edges = np.histogram(synaptic_input.ravel(), density=True, bins=bins)
        hist_inputs[t] = H    
    
    edge_mids = (edges[:-1] + edges[1:]) / 2
    ax.plot(edge_mids, hist_inputs.mean(axis=0), color="tab:blue", label="recurrent")
    
    ax.bar(edges[:-1], hist_inputs.mean(axis=0), #yerr=hist_inputs.std(axis=0), 
           align="edge", width=0.8*(edges[1]-edges[0]), fill=False, edgecolor="tab:blue",
           #capsize=4, error_kw={"elinewidth": 2},
           # label="recurrent", 
           log=True, alpha=.4,
    )
    
    # ext. input
    # We can just take the std here, as the noise is defined by n_dot = -n/tau + sigma*sqrt(2/tau)*GWN
    # And for such an OU process, the std in stationarity is defined as sqrt(tau/2)*std of the OU process which has the inverse prefactor.
    drive = np.random.normal(config.drive.mean, config.drive.std, size=100000)
    D, edges = np.histogram(drive, density=True, bins=bins) 
    
    edge_mids = (edges[:-1] + edges[1:]) / 2
    ax.plot(edge_mids, D, color="tab:orange", label="external")
    
    ax.bar(edges[:-1], D, #log=True,
           align="edge", width=0.8*(edges[1]-edges[0]), fill=False, edgecolor="tab:orange",
           # capsize=4, error_kw={"elinewidth": 2},
           # label="external", 
           log=True, alpha=.4,
    )
    
    ax.axvline(syn_inputs.mean(), ls="dotted", c="k", label="mean (rec.)", ymax=0.8)
    ax.axvline(config.transfer_function.offset, ls="--", c="b", label="half-activation", ymax=0.8)

    # Add the fraction of inputs that lead to a rate output of 0.5
    portion = hist_inputs[:, edges[:-1] >= SYN_THRESHOLD].sum() / hist_inputs.sum()        
    #ax.text(x=100, y=D.max(), s=f"p={portion:.2%}\nas spikes")
        
    ax.set_yscale('log')
    ax.set_yticks((1e-1, 1e-3, 1e-5, 1e-7))
    ax.legend(
        loc="upper center",
        # handlelength = 1.4,
        # borderpad = 0.2,
        # labelspacing = 0.1,
        columnspacing=1, 
        ncols=2,
        # handletextpad=0.4,
        # reverse=True,
    )
    

#===============================================================================
# SCATTER TARGETS
#===============================================================================


def scatter_targets(ax:object, shift:float, center:np.ndarray, std:float, n_conn:int, **plot_kwargs):
    """Finds targets and scatter them."""
    targets = np.random.normal(scale=std, size=(n_conn, 2))
    ax.plot(*(targets + center + shift).T, **plot_kwargs)


#===============================================================================
# HISTORGRAM DISTRIBUTIONS
#===============================================================================

def hist_exc_dist(ax:object, shift:int, axis:str, std:float, n_conn:int, center=(0, 0), **style):
    logger.info("Hist. of exc. distribution")
    dist, bins = get_hist_of_normal(center + shift, std, size=n_conn)
    handle = hist_dist(ax, bins, dist, axis=axis, **style)
    return dist, bins, handle


def hist_inh_dist(ax:object, axis:str, std:float, n_conn:int, center=(0, 0), factor:int=2., **style):
    logger.info("Hist. of inh. distribution")
    dist, bins = get_hist_of_normal(center, std * factor, size=n_conn)
    dist /= factor
    handle = hist_dist(ax, bins, dist, axis=axis, invert=True, **style)
    return dist, bins, handle


def hist_dist(ax:object, bins, dist, axis:str="x", invert:bool=False, **style):
    """
    Plotting a histogram either on the x or on the y axis.
    """
    if invert:
        dist = -dist
    offset = 4 # Required to have a nicer plot
    dist = dist + offset
    if axis == "x":
        data = bins[1:], dist
    elif axis == "y":
        data = dist, bins[:-1]
    return ax.step(*data, **style)


def get_hist_of_normal(mean, std, size, normalize:bool=True):
    """
    Histograms a normal distribution in two dimensions.
    """
    dist = np.random.normal(mean, std, size=(size, 2))
    bins = np.arange(0, nrows+1, BIN_WIDTH)
    dist_hist, _ = np.histogram(dist, bins)
    if normalize:
        dist_hist = normalize_histogram(dist_hist)
    return dist_hist, bins



def normalize_histogram(distribution, prefactor=MAX_HIST):
    """Normalizes a histogram {distribution} such that the maximum value is the {prefactor}."""
    return prefactor * distribution / distribution.max()


def plot_scalebar(x:float, y:float, width:float, **scalebar_style):
    plt.plot([x, x+width], [y, y], **scalebar_style)

#===============================================================================
# UTIL
#===============================================================================

def add_break_marks(axes:tuple, d:float=0.02):
    kwargs = dict(transform=axes[0].transAxes, color="k", clip_on=False)
    axes[0].plot((1-d, 1+d), (-d, +d), **kwargs)
    # axes[0].plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs = dict(transform=axes[1].transAxes, color="k", clip_on=False)
    axes[1].plot((-d, +d), (-d, +d), **kwargs)
    # axes[1].plot((-d, +d), (1-d, 1+d), **kwargs)



if __name__ == '__main__':
    main()
    plt.show()
