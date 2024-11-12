#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""
import numpy as np
import matplotlib.pyplot as plt
from lib import functimer

from .basic import create_horizontal_slider, create_vertical_slider

from plot.constants import NORM_ACTIVITY, COLOR_MAP_ACTIVITY


def plot_activity(data:np.ndarray, title:str=None, figname:str=None, norm:tuple=None, cmap=None, figsize=None):
    figsize = figsize or (4, 3)
    fig = plt.figure(figname, figsize=figsize)
    create_image(data, norm, cmap)
    plt.title(title)
    plt.colorbar()
    return fig


def get_width(size:int):
    return int(np.sqrt(size))

@functimer
def create_image(data:np.ndarray, norm:tuple=None, cmap=None, axis:object=None):
    """
    Creates an image from a flat data (reshapes it to a square).

    'norm' and 'cmap' are optional.
    If axis is specified, the image is shown in that axis.
    """
    norm = norm or NORM_ACTIVITY
    cmap = cmap or COLOR_MAP_ACTIVITY
    # If not provided take the general plt-method.
    ax = axis if axis is not None else plt

    width = get_width(data.size)
    return ax.imshow(data.reshape((width, width)), origin="lower", vmin=norm[0], vmax=norm[1], cmap=cmap)


def image_slider_2d(data:np.ndarray, fig:object, axis:object, label:str="index", **image_kwargs):

    def __update_activity(val):
        idx_pre = int(slider_hor.val)
        idx_post = int(slider_ver.val)
        axis.cla()
        create_image(data[idx_pre, idx_post], axis=axis, **image_kwargs)
        axis.set_title(f"Comparison of {label} {idx_pre} (pre) and {idx_post} (post)")
        fig.canvas.draw_idle()

    slider_hor = create_horizontal_slider(data_length=len(data), on_change=__update_activity, label=label)
    slider_ver = create_vertical_slider(data_length=len(data), on_change=__update_activity, label=label)
    # Initialize the plot.
    slider_hor.set_val(slider_hor.val)
    return slider_hor, slider_ver


def image_slider_1d(data:np.ndarray, fig:object, axis:object, method, label:str="Index"):

    def __update_activity(val):
        idx = int(slider_hor.val)
        axis.cla()
        method(idx=idx)
        fig.canvas.draw_idle()

    slider_hor = create_horizontal_slider(data_length=len(data), on_change=__update_activity, label=label)
    # Initialize the plot.
    slider_hor.set_val(slider_hor.val)
    return slider_hor


def _image_slider_1d(data:np.ndarray, fig:object, axis:object, label:str="index", **image_kwargs):

    def __update_activity(val):
        idx = int(slider_hor.val)
        axis.cla()
        create_image(data[idx], axis=axis, **image_kwargs)
        axis.set_title(f"Display of {label} {idx}")
        fig.canvas.draw_idle()

    slider_hor = create_horizontal_slider(data_length=len(data), on_change=__update_activity, label=label)
    # Initialize the plot.
    slider_hor.set_val(slider_hor.val)
    return slider_hor
