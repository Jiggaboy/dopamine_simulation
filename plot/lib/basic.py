#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Has the parts General, Circles, Slider, Colorbar

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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import lib.universal as UNI

from plot.constants import min_degree, max_degree, CMAP_DEGREE

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

def add_topright_spines(ax:object):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(True)

#===============================================================================
# PATCHES/CIRCLES
#===============================================================================

def plot_patch(center:tuple, radius:int, width:int, add_outline=True, **kwargs)->None:
    kwargs["ec"] = kwargs.get("ec", "black")
    kwargs["ls"] = kwargs.get("ls", "solid")
    kwargs["lw"] = kwargs.get("lw", 1.6)
    kwargs["zorder"] = kwargs.get("zorder", 10)

    kwargs_background = dict(kwargs)
    kwargs_background["ec"] = "black"
    kwargs_background["lw"] = 2.25 if add_outline else 0

    center = np.asarray(center)
    
    # Plot the circle on location
    draw_circle(center, radius=radius, **kwargs_background)
    draw_circle(center, radius=radius, **kwargs)

    # Plot the circle on the other side of the toroid
    for idx, c in enumerate(center):
        if c + radius > width:
            n_center = center.copy()
            n_center[idx] = n_center[idx] - width
            draw_circle(n_center, radius=radius, **kwargs_background)
            draw_circle(n_center, radius=radius, **kwargs)
        if c - radius < 0:
            n_center = center.copy()
            n_center[idx] = n_center[idx] + width
            draw_circle(n_center, radius=radius, **kwargs_background)
            draw_circle(n_center, radius=radius, **kwargs)
    # Plot it also, when both sides are exceeded
    if all(center + radius > width):
        n_center = center.copy() - width
        draw_circle(n_center, radius=radius, **kwargs_background)
        draw_circle(n_center, radius=radius, **kwargs)


def draw_circle(center, radius, **kwargs):
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


def map_indegree_to_color(indegree:float, min_degree:float=min_degree, max_degree:float=max_degree) -> float:
    indegree = min_degree if indegree < min_degree else indegree
    indegree = max_degree if indegree > max_degree else indegree
    color = CMAP_DEGREE((indegree - min_degree) / (max_degree - min_degree))
    return color
