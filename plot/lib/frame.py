#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:36:16 2022

@author: Hauke Wernecke
"""
import numpy as np
import matplotlib.pyplot as plt

from .basic import create_horizontal_slider, create_vertical_slider


COLOR_MAP = plt.cm.hot_r


def plot_activity(data:np.ndarray, title:str=None, figname:str=None, norm:tuple=None, cmap=None, figsize=None):
    figsize = figsize or (4, 3)
    plt.figure(figname, figsize=figsize)
    create_image(data, norm, cmap)
    plt.title(title)
    plt.colorbar()


def create_image(data:np.ndarray, norm:tuple=None, cmap=None, axis:object=None):
    norm = norm or (0, 1)
    cmap = cmap or COLOR_MAP

    width = int(np.sqrt(data.size))
    if axis is not None:
        axis.imshow(data.reshape((width, width)), origin="lower", vmin=norm[0], vmax=norm[1], cmap=cmap)
        return
    plt.imshow(data.reshape((width, width)), origin="lower", vmin=norm[0], vmax=norm[1], cmap=cmap)

    
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