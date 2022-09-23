#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:08:08 2022

@author: hauke
"""

import cflogger
logger = cflogger.getLogger()

import numpy as np
import matplotlib.pyplot as plt
from util import pickler as PIC


logger.info("Start: Preparing figure of the connectivity distribution")

nrows = 60
MARKER = "."
MAX_HIST = 12
BIN_WIDTH = 1

FIGSIZE = (4.2, 4.2)
SAVE_FIG_PATH = "./figures/{}.svg"
AXIS_MARGIN = 5

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
c_unshifted = color_cycle[0]
c_shifted = color_cycle[2]


KTH_GREEN = np.asarray((176, 201, 43)) / 255
KTH_PINK = np.asarray((216, 84, 151)) / 255
KTH_GREY = np.asarray((101, 101, 108)) / 255
KTH_BLUE = np.asarray((25, 84, 166)) / 255

#from plot import KTH_GREEN, KTH_PINK, KTH_GREY, KTH_BLUE


colors = KTH_GREEN, KTH_PINK, KTH_GREY, KTH_BLUE

c_exc = colors[0]
c_inh = colors[1]
c_mexican = colors[2]
c_neuron = "red"

NEURON_SIZE = 6
SPINE_WIDTH = 4

C_TARGET = KTH_GREEN
C_TARGET_SHIFTED = KTH_BLUE

#### DISTRIBUTION
CENTER = np.asarray((30, 30), dtype=int)
STD = 4
SHIFT = 5
N_CONN = 300


#### STYLE HISTOGRAMS
C_EXC_HIST = KTH_PINK
C_INH_HIST = KTH_PINK
C_FULL_HIST = KTH_GREY
LW_DIST = 2

#### JOINT CONNECTIVITY
joint_connectivity_fig_params = {
    "num": "joint_connectivity", 
    "figsize": (3.4, 3)
}




def main():

    joint_connectivity()
    plt.show()
    
    quit()
    ## plot shifted and unshifted seperately
    titles = ("unshifted", "shifted")
    for t, s in zip(titles, shifts):
        plt.figure(num=t, figsize=FIGSIZE)
        plt.title("Connectivity distribution")
        scatter_targets(std, size=n_conn, shift=s)
        hist_connectitivy_distributions(std, shift=shifts)

        plot_scalebar()
        set_layout()
        plt.savefig(SAVE_FIG_PATH.format(t))
        
        
def joint_connectivity():
    shifts = (0, SHIFT)
    
    fig = joint_connectivity_figure()
    neuron_handle = scatter_all_targets()
    exc_handle_unshifted, inh_handle, full_dist_handle = hist_full_dist(shift=0, axis="x")
    exc_handle_shifted, _, _ = hist_full_dist(shift=SHIFT, axis="y")
    plt.legend([*full_dist_handle, *exc_handle_shifted, *exc_handle_unshifted, *inh_handle, *neuron_handle], 
              ["joint", "exc. (shifted)", "exc. (unshifted)", "inh.", "pre-syn. neuron"])
    PIC.save_figure(fig.get_label(), fig)
    
    
def joint_connectivity_figure():
    fig = plt.figure(**joint_connectivity_fig_params)
    plt.title("Connectivity distributions")
    plot_scalebar()
    set_layout(spine_width=0)
    xylabel()
    return fig
    
    
        
def plot_scalebar():
    x, y = 14, 55
    plt.plot([x, x+2], [y, y], color="black", linewidth=2)


def hist_connectitivy_distributions(shift:float=0, position:np.ndarray=CENTER, std:float=STD, size:int=N_CONN):
    axes = ("x", "y")

    for pos, (s, axis) in enumerate(zip(shift, axes)):
        exc_dist, inh_dist, bins = get_exc_inh_distributions(position[pos], std, size=size, shift=s)
        mexican_hat = exc_dist - inh_dist
        plot_hist_dist(bins[:-1], exc_dist, inh_dist, mexican_hat, axis=axis)

        
############# SCATTER TARGETS ###########################################################################################
        
def scatter_all_targets(shift:float=SHIFT):
    logger.info("Scatter all targets")
    scatter_targets(shift=0, color=C_TARGET)
    scatter_targets(shift=SHIFT, color=C_TARGET_SHIFTED)
    neuron_handle = plot_neuron()
    return neuron_handle
    

def scatter_targets(shift:float=0, std:float=STD, size:int=N_CONN, color=C_TARGET):
    targets = np.random.normal(scale=std, size=(size, 2))
    plt.plot(*(targets + CENTER + shift).T, color=color, marker=MARKER, linestyle="None", ms=4)

        
############# HISTOGRAM DISTRIBUTIONS ###################################################################################


def plot_hist_dist(bins, exc_dist, inh_dist, full_dist=None, axis:str="x"):
    colors = (c_exc, c_inh, c_mexican)
    dists = [exc_dist, inh_dist]
    
    
    if full_dist is not None:
        dists.append(full_dist)

    lw = 1
    if axis == "x":
        for dist, color in zip(dists, colors):
            if (color == colors[-1]).all():
                lw = 3
            plt.step(bins, dist, color=color, lw=lw)
    elif axis == "y":
        for dist, color in zip(dists, colors):
            if (color == colors[-1]).all():
                lw = 3
            plt.step(dist, bins, color=color, lw=lw)
            

def hist_exc_dist(shift:int=0, axis:str="x", std:float=STD, size=100*N_CONN, center=CENTER, **style):
    logger.info("Hist. of exc. distribution")
    dist, bins = get_hist_of_normal(center + shift, std, size=size)
    handle = hist_dist(bins, dist, axis=axis, **style)
    return dist, bins, handle
            

def hist_inh_dist(axis:str="x", std:float=STD, size=100*N_CONN, center=CENTER, **style):
    logger.info("Hist. of inh. distribution")
    dist, bins = get_hist_of_normal(center, std * 2, size=size)
    dist /= 2
    handle = hist_dist(bins, dist, axis=axis, **style)
    return dist, bins, handle
            

def hist_full_dist(shift:int=0, axis:str="x", std:float=STD, size=100*N_CONN, center=CENTER):
    logger.info(f"Histogram distribution with shift {shift}.")
    exc_color = C_TARGET_SHIFTED if shift else C_TARGET
    exc_dist, bins, exc_handle = hist_exc_dist(shift=shift, axis=axis, color=exc_color)
    inh_dist, bins, inh_handle = hist_inh_dist(axis=axis, color=C_INH_HIST)
    mexican_hat = exc_dist - inh_dist
    full_dist_handle = hist_dist(bins, mexican_hat, axis=axis, lw=LW_DIST, color=C_FULL_HIST)
    return exc_handle, inh_handle, full_dist_handle

    
def hist_dist(bins, dist, axis:str="x", **style):
    """
    Plotting a histogram either on the x or on the y axis.
    """
    if axis == "x":
        data = bins[:-1], dist
    elif axis == "y":
        data = dist, bins[:-1]
    return plt.step(*data, **style)
        
        
########################################################################################################################
        

def get_exc_inh_distributions(position, std:float, size:int, shift=0):
    """
    Retrieves two normalized Gauss distributions.
    The exc. distribution has a std of {std}, whereas the inh. distribution has a std of {2 * std}, but normalizes to the half.
    """
    exc_dist, bins = get_hist_of_normal(position + shift, std, size=size)
    inh_dist, bins = get_hist_of_normal(position, 2*std, size=size)
    # Scale the inh. distribution
    inh_dist = inh_dist / 2
    return exc_dist, inh_dist, bins


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


def plot_neuron():
    return plt.plot(*CENTER, marker="o", ms=NEURON_SIZE, ls="None", color=c_neuron)
    return plt.plot(*CENTER, marker="o", ms=NEURON_SIZE, ls="None", mec=c_neuron, mew=3, mfc="white")


def normalize_histogram(distribution, prefactor=MAX_HIST):
    """Normalizes a histogram {distribution} such that the maximum value is the {prefactor}."""
    return MAX_HIST * distribution / distribution.max()


def set_layout(rows:int=nrows, margin:float=AXIS_MARGIN, spine_width:float=SPINE_WIDTH):
    ax = plt.gca()

    PY_OFFSET = -.5
    lim = PY_OFFSET - margin, PY_OFFSET + rows + margin
    plt.xlim(*lim)
    plt.ylim(*lim)
    ticks = np.linspace(0, rows, 3, endpoint=True)
    #ax.set_xticks(ticks)
    #ax.set_yticks(ticks)
    ax.set_xticks([])
    ax.set_yticks([])

    tick_params = {"width": spine_width, "length": spine_width * 3, "labelleft": False, "labelbottom": False}
    ax.tick_params(**tick_params)

    for s in ('top', 'right'):
        #ax.spines[s].set_visible(False)
        ax.spines[s].set_linewidth(spine_width)
    for s in ('bottom', 'left'):
        #ax.spines[s].set_visible(False)
        ax.spines[s].set_linewidth(spine_width)

    # plt.tight_layout()
    
    
def xylabel():   
    plt.xlabel("unshifted (symmetric)")
    plt.ylabel("shifted (asymmetric)")
    
    
if __name__ == "__main__":
    main()
