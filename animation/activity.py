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



def activity(data:np.ndarray, nrows:int, title:str=None, figname:str=None, norm:tuple=(0, 1)):
    plt.figure(figname)
    plt.title(title)
    plt.imshow(data.reshape((nrows, nrows)), origin="lower", vmin=norm[0], vmax=norm[1])
    plt.colorbar()


def animate_firing_rates(rate:np.ndarray, coordinates:np.ndarray, maxNeurons:int=1, **animparams):
    interval = animparams.get("interval", 200)
    start = animparams.get("start", 10)
    stop = animparams.get("stop", rate.shape[1])
    step = animparams.get("step", 5)

    FIG_NAME = "firing_rate_animation"
    fig = plt.figure(FIG_NAME)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=rate[:, CF.WARMUP:].max())
    image = plt.imshow(rate[:maxNeurons, 1].reshape((CF.SPACE_WIDTH, CF.SPACE_WIDTH)), cmap=COLOR_MAP, norm=norm, origin="lower")
    plt.colorbar(image)
    def animate(i):
        plt.figure(FIG_NAME)
        image.set_data(image.to_rgba(rate[:maxNeurons, i].reshape((CF.SPACE_WIDTH, CF.SPACE_WIDTH))))
        plt.title(f"Time point: {i}")

    return FuncAnimation(fig, animate, interval=interval, frames=range(start, stop, step))
