#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Has the parts General, Circles, Slider, Colorbar

@author: Hauke Wernecke
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm, colors
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import lib.universal as UNI




#===============================================================================
# GENERAL
#===============================================================================

def remove_spines_and_ticks(ax:object):
    # remove_spines(ax)
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)

    # remove_ticks(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    

def remove_topright_spines(ax:object):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)



#===============================================================================
# PATCHES/CIRCLES
#===============================================================================

def plot_patch(center:tuple, radius:int, width:int, **kwargs)->None:
    if not kwargs.get("ec"):
        kwargs["ec"] = "black"
    if not kwargs.get("ls"):
        kwargs["ls"] = "solid"
    if not kwargs.get("zorder"):
        kwargs["zorder"] = 10

    center = np.asarray(center)
    # Plot the circle on location
    black_dashed_circle(center, radius=radius, **kwargs)

    # Plot the circle on the other side of the toroid
    for idx, c in enumerate(center):
        if c + radius > width:
            n_center = center.copy()
            n_center[idx] = n_center[idx] - width
            black_dashed_circle(n_center, radius=radius, **kwargs)
        if c - radius < 0:
            n_center = center.copy()
            n_center[idx] = n_center[idx] + width
            black_dashed_circle(n_center, radius=radius, **kwargs)
    # Plot it also, when both sides are exceeded
    if all(center + radius > width):
        n_center = center.copy() - width
        black_dashed_circle(n_center, radius=radius, **kwargs)


def black_dashed_circle(center, radius, **kwargs):
    ax = kwargs.get("axis")
    kwargs.pop("axis", None)
    circle = mpatches.Circle(center, radius=radius, fc="None", linewidth=2, **kwargs)
    if ax:
        ax.add_artist(circle)
    else:
        plt.gca().add_artist(circle)


def plot_patch_from_tag(tag:str, config:object, **kwargs):
    name = UNI.name_from_tag(tag)
    center = config.get_center(name)
    # percent = UNI.split_percentage_from_tag(tag)
    #
    # ec, alpha = get_color(float(percent) / 100)
    # kwargs["ec"] = ec
    # kwargs["alpha"] = alpha

    radius =  UNI.radius_from_tag(tag)
    if np.asarray(center).size > 2:
        for c in center:
            plot_patch(c, float(radius), width=config.rows, **kwargs)
    else:
        plot_patch(center, float(radius), width=config.rows, **kwargs)

def get_color(percent:float, maxpercent:float=0.2, poscolor="green", negcolor="red")->tuple:
    """Return color and alpha value."""
    color = "red" if percent < 0 else "green"
    return color, abs(percent / maxpercent) 
        
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

def add_colorbar(axis:object, norm:tuple, cmap:object, **plot_kwargs):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    normalize = colors.Normalize(*norm)
    cbar = plt.colorbar(
        cm.ScalarMappable(norm=normalize, cmap=cmap),
        cax=cax,
        **plot_kwargs
    )
    return cbar

def add_colorbar_from_im(axis:object, im:object) -> object:
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    cbar = plt.colorbar(im,
        orientation="vertical",
        cax=cax
    )
    return cbar
