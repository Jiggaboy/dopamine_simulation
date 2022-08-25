#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2022-07-11

@author: Hauke Wernecke
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider

import numpy as np
    
        

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