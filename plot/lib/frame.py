#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:36:16 2022

@author: hauke
"""
import numpy as np
import matplotlib.pyplot as plt

COLOR_MAP = plt.cm.hot_r


def plot_activity(data:np.ndarray, title:str=None, figname:str=None, norm:tuple=None, cmap=None, figsize=None):
    figsize = figsize or (4, 3)
    plt.figure(figname, figsize=figsize)
    create_image(data, norm, cmap)
    plt.title(title)
    plt.colorbar()


def create_image(data:np.ndarray, norm:tuple=None, cmap=None):
    norm = norm or (0, 1)
    cmap = cmap or COLOR_MAP

    width = int(np.sqrt(data.size))
    plt.imshow(data.reshape((width, width)), origin="lower", vmin=norm[0], vmax=norm[1], cmap=cmap)
