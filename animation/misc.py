#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:25:22 2021

@author: hauke
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import configuration as CF


CIRCLE_COLOR = "black"



def plot_patch(center:tuple, radius:int)->None:
    plot_circle(center, radius=radius)
    center = np.asarray(center)
    for idx, c in enumerate(center):
        if c + radius > CF.SPACE_WIDTH:
            n_center = center.copy()
            n_center[idx] = n_center[idx] - CF.SPACE_WIDTH
            plot_circle(n_center, radius=radius)
    if all(center + radius > CF.SPACE_WIDTH):
        n_center = center.copy() - CF.SPACE_WIDTH
        plot_circle(n_center, radius=radius)


def plot_circle(center, radius):
    circle = mpatches.Circle(center, radius=radius, fc="None", ec=CIRCLE_COLOR, linewidth=2, ls="dashed")
    plt.gca().add_artist(circle)
