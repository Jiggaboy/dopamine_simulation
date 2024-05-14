#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2022-07-11

@author: Hauke Wernecke
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colors
from matplotlib.widgets import Slider

import numpy as np




#===============================================================================
# GENERAL
#===============================================================================

def set_layout(ax:object, rows:int, margin:float, spine_width:float):
    PY_OFFSET = -.5
    lim = PY_OFFSET - margin, PY_OFFSET + rows + margin
    ticks = np.linspace(0, rows, 3, endpoint=True)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    tick_params = {
        "width": spine_width,
        "length": spine_width * 3,
        "labelleft": False,
        "labelbottom": False}
    ax.tick_params(**tick_params)

    for s in ('top', 'right'):
        ax.spines[s].set_linewidth(spine_width)
    for s in ('bottom', 'left'):
        ax.spines[s].set_linewidth(spine_width)
    # plt.tight_layout()


def remove_spines_and_ticks(ax:object):
    remove_spines(ax)
    remove_ticks(ax)


def remove_spines(ax:object):
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)


def remove_ticks(ax:object):
    ax.set_xticks([])
    ax.set_yticks([])




############ To be checked #############################################################################################
def bold_spines(ax, width:float=1):
    tick_params = {"width": width, "length": width * 3, "labelleft": True, "labelbottom": True}
    ax.tick_params(**tick_params)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('bottom', 'left'):
            ax.spines[s].set_linewidth(width)
########################################################################################################################

#===============================================================================
# PATCHES/CIRCLES
#===============================================================================

def plot_patch(center:tuple, radius:int, width:int)->None:
    # Plot the circle on location
    black_dashed_circle(center, radius=radius)

    # Plot the circle on the other side of the toroid
    center = np.asarray(center)
    for idx, c in enumerate(center):
        if c + radius > width:
            n_center = center.copy()
            n_center[idx] = n_center[idx] - width
            black_dashed_circle(n_center, radius=radius)
        if c - radius < 0:
            n_center = center.copy()
            n_center[idx] = n_center[idx] + width
            black_dashed_circle(n_center, radius=radius)
    # Plot it also, when both sides are exceeded
    if all(center + radius > width):
        n_center = center.copy() - width
        black_dashed_circle(n_center, radius=radius)


def white_dashed_circle(center:tuple, radius:float)->None:
    circle = mpatches.Circle(center, radius=radius, fc="None", ec="white", linewidth=2, ls="dashed")
    plt.gca().add_artist(circle)


def black_dashed_circle(center, radius):
    circle = mpatches.Circle(center, radius=radius, fc="None", ec="black", linewidth=2, ls="dashed")
    plt.gca().add_artist(circle)


#===============================================================================
# SLIDER
#===============================================================================

def create_horizontal_slider(data_length:int, on_change:callable, label:str)->object:
    ax = plt.axes([0.25, 0.025, 0.55, 0.03])
    slider = Slider(ax, label, valmin=0, valmax=data_length - 1, valfmt='%d', valstep=range(data_length))
    slider.on_changed(on_change)
    return slider


def create_vertical_slider(data_length:int, on_change:callable, label:str)->object:
    ax = plt.axes([0.1, 0.25, 0.03, 0.65])
    slider = Slider(ax, label, valmin=0, valmax=data_length - 1, valinit=0, valfmt='%d', valstep=range(data_length), orientation="vertical")
    slider.on_changed(on_change)
    return slider

#===============================================================================
# COLORBAR
#===============================================================================

def add_colorbar(axis:object, norm:tuple, cmap:object):
    normalize = colors.Normalize(*norm)
    cb = plt.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), ax=axis, ticks=np.linspace(*norm, 3),
                  # fraction=.04,
                  # orientation="horizontal"
                  )
    return cb
