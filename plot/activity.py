#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from plot.lib import add_colorbar
from lib import functimer

import lib.universal as UNI

# hot
from plot.constants import COLOR_MAP_ACTIVITY
COLOR_MAP_DIFFERENCE = plt.cm.seismic
COLOR_MAP_DEFAULT = COLOR_MAP_ACTIVITY

from plot import NORM_ACTIVITY
NORM_DEFAULT = NORM_ACTIVITY


def get_width(size:int):
    return int(np.sqrt(size))


@functimer
def create_image(data:np.ndarray, norm:tuple=None, cmap=None, axis:object=None):
    """
    Creates an image from a flat data (reshapes it to a square).

    'norm' and 'cmap' are optional.
    If axis is specified, the image is shown in that axis.
    """
    norm = norm or NORM_DEFAULT
    cmap = cmap or COLOR_MAP_DEFAULT
    # If not provided take the general plt-method.
    ax = axis if axis is not None else plt

    width = get_width(data.size)
    return ax.imshow(data.reshape((width, width)), origin="lower", vmin=norm[0], vmax=norm[1], cmap=cmap)


def activity(*data:np.ndarray, title:str=None, figname:str=None, norm:tuple=None, cmap=None, figsize=None, ax_titles:list=None):
    norm = norm or NORM_DEFAULT
    cmap = cmap or COLOR_MAP_DEFAULT
    figsize = figsize or (4, 3)
    fig, axes = plt.subplots(ncols=len(data), num=figname, figsize=figsize)
    axes = UNI.make_iterable(axes)

    for idx, (ax, d) in enumerate(zip(axes, data)):
        create_image(d, norm, cmap, axis=ax)
        add_colorbar(ax, norm, cmap)
        try:
            ax.set_title(ax_titles[idx])
        except Exception:
            pass
    fig.suptitle(title)
    return fig


def pre_post_activity(pre:np.ndarray, post:np.ndarray, **descriptors):
    figname = descriptors.get("figname")
    fig, axes = plt.subplots(ncols=2, sharey=True, num=figname)
    # pre
    plt.sca(axes[0])
    create_image(pre)
    title_pre = descriptors.get("title_pre")
    plt.title(title_pre)

    # post
    plt.sca(axes[1])
    create_image(post)
    title_post = descriptors.get("title_post")
    plt.title(title_post)


def animate_firing_rates(fig:object, method:callable, **animparams):
    interval = animparams.get("interval", 200)
    start = animparams.get("start", 0)
    stop = animparams.get("stop", 1000)
    step = animparams.get("step", 10)

    return FuncAnimation(fig, method, interval=interval, frames=range(start, stop, step), cache_frame_data=False)
