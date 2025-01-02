#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""

import numpy as np
import matplotlib.pyplot as plt

from plot.lib import add_colorbar

import lib.universal as UNI

# hot
from plot.lib.frame import create_image
from plot.constants import COLOR_MAP_ACTIVITY
COLOR_MAP_DIFFERENCE = plt.cm.seismic
COLOR_MAP_DEFAULT = COLOR_MAP_ACTIVITY

from plot import NORM_ACTIVITY
NORM_DEFAULT = NORM_ACTIVITY


# MOVED TO PLOT.LIB.FRAME
# def get_width(size:int):
#     return int(np.sqrt(size))

# MOVED TO PLOT.LIB.FRAME
# @functimer
# def create_image(data:np.ndarray, norm:tuple=None, cmap=None, axis:object=None):
#     """
#     Creates an image from a flat data (reshapes it to a square).

#     'norm' and 'cmap' are optional.
#     If axis is specified, the image is shown in that axis.
#     """
#     norm = norm or NORM_DEFAULT
#     cmap = cmap or COLOR_MAP_DEFAULT
#     # If not provided take the general plt-method.
#     ax = axis if axis is not None else plt

#     width = get_width(data.size)
#     return ax.imshow(data.reshape((width, width)), origin="lower", vmin=norm[0], vmax=norm[1], cmap=cmap)


def activity(*data:np.ndarray, title:str=None, figname:str=None, norm:tuple=None, cmap=None, figsize=None):
    norm = norm or NORM_DEFAULT
    cmap = cmap or COLOR_MAP_DEFAULT
    figsize = figsize or (4, 3)
    fig, axes = plt.subplots(ncols=len(data), num=figname, figsize=figsize)
    axes = UNI.make_iterable(axes)

    for idx, (ax, d) in enumerate(zip(axes, data)):
        create_image(d, norm, cmap, axis=ax)
        add_colorbar(ax, norm, cmap)
    fig.suptitle(title)
    return fig
