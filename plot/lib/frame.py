#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""
import numpy as np
import matplotlib.pyplot as plt
from lib import functimer
import lib.universal as UNI

from .basic import add_colorbar

from plot.constants import NORM_ACTIVITY, COLOR_MAP_ACTIVITY


def plot_activity(data:np.ndarray, title:str=None, figname:str=None, norm:tuple=None, cmap=None, figsize=None):
    figsize = figsize or (4, 3)
    fig = plt.figure(figname, figsize=figsize)
    create_image(data, norm, cmap)
    plt.title(title)
    plt.colorbar()
    return fig


def create_images_on_axes(axes:object, data:np.ndarray, norm:tuple, cmap, with_colorbar:bool=True):
    axes = UNI.make_iterable(axes)
    for idx, (ax, d) in enumerate(zip(axes, data)):
        create_image(d, norm, cmap, axis=ax)
        if with_colorbar:
            add_colorbar(ax, norm, cmap)


def get_width(size:int):
    return int(np.sqrt(size))

# @functimer
def create_image(data:np.ndarray, norm:tuple=None, cmap=None, axis:object=None):
    """
    Creates an image from a flat data (reshapes it to a square).

    'norm' and 'cmap' are optional.
    If axis is specified, the image is shown in that axis.
    """
    norm = norm or (data.min(), data.max())
    cmap = cmap or COLOR_MAP_ACTIVITY
    # If not provided take the general plt-method.
    ax = axis if axis is not None else plt

    width = get_width(data.size)
    return ax.imshow(data.reshape((width, width)), origin="lower", vmin=norm[0], vmax=norm[1], cmap=cmap)

