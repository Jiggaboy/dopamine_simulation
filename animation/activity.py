#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:04:00 2021

@author: hauke
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# hot
COLOR_MAP_ACTIVITY = plt.cm.hot_r
COLOR_MAP_DIFFERENCE = plt.cm.seismic
COLOR_MAP_DEFAULT = COLOR_MAP_ACTIVITY

NORM_DIFFERENCE = -.5, .5
NORM_ACTIVITY = 0, .5
NORM_DEFAULT = NORM_ACTIVITY

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
    
    width = int(np.sqrt(data.size))
    return ax.imshow(data.reshape((width, width)), origin="lower", vmin=norm[0], vmax=norm[1], cmap=cmap)


def activity(data:np.ndarray, title:str=None, figname:str=None, norm:tuple=None, cmap=None, figsize=None):
    figsize = figsize or (4, 3)
    plt.figure(figname, figsize=figsize)
    create_image(data, norm, cmap)
    plt.title(title)
    plt.colorbar()


def pre_post_activity(pre:np.ndarray, post:np.ndarray, **descriptors):
    figname = descriptors.get("figname")
    fig, axes = plt.subplots(ncols=2, sharey=True, num=figname)
    # pre
    plt.sca(axes[0])
    create_image(pre)
    title_pre = descriptors.get("title_pre")
    plt.title(title_pre)
    # plt.colorbar()
    # cbar = plt.colorbar()

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

    return FuncAnimation(fig, method, interval=interval, frames=range(start, stop, step))
