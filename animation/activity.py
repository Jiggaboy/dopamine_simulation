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

import configuration as CF

COLOR_MAP = plt.cm.hot


def create_image(data:np.ndarray, norm:tuple=None, cmap=None):
    norm = norm or (0, 1)
    cmap = cmap or plt.cm.jet

    width = int(np.sqrt(data.size))
    plt.imshow(data.reshape((width, width)), origin="lower", vmin=norm[0], vmax=norm[1], cmap=cmap)


def activity(data:np.ndarray, title:str=None, figname:str=None, norm:tuple=None, cmap=None):
    plt.figure(figname, figsize=(4, 3))
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
    # plt.colorbar()
    # cbar.remove()





def animate_firing_rates(rate:np.ndarray, coordinates:np.ndarray, maxNeurons:int=1, **animparams):
    interval = animparams.get("interval", 200)
    start = animparams.get("start", 10)
    stop = animparams.get("stop", rate.shape[1])
    step = animparams.get("step", 5)

    FIG_NAME = "firing_rate_animation"
    fig = plt.figure(FIG_NAME, figsize=(4, 3))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    image = plt.imshow(rate[:maxNeurons, 1].reshape((CF.SPACE_WIDTH, CF.SPACE_WIDTH)), cmap=COLOR_MAP, norm=norm, origin="lower")
    plt.title("Snapshot of ongoing activity")
    plt.colorbar(image)
    def animate(i):
        plt.figure(FIG_NAME)
        image.set_data(image.to_rgba(rate[:maxNeurons, i].reshape((CF.SPACE_WIDTH, CF.SPACE_WIDTH))))
        # plt.title(f"Time point: {i}")

    return FuncAnimation(fig, animate, interval=interval, frames=range(start, stop, step))
