#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Plots the position of excitatory and inhibitory neurons.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.3'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

from plot.constants import KTH_GREEN, KTH_PINK, KTH_GREY, KTH_BLUE

#===============================================================================
# CONSTANTS
#===============================================================================

rcParams["font.size"] = 12
rcParams["figure.figsize"] = (3.5, 3.5)

name = "EI_layout"
directory = "figures"
side_length = 6
pos = np.arange(side_length)
x_pos_exc, y_pos_exc = np.meshgrid(pos, pos)


iside = np.arange(0.5, side_length, 2)
x_pos_inh, y_pos_inh = np.meshgrid(iside, iside)

fig, ax = plt.subplots(num=name)

ax.scatter(x_pos_exc, y_pos_exc, color=KTH_PINK, marker="o", label="exc. neuron")
ax.scatter(x_pos_inh, y_pos_inh, color=KTH_BLUE, marker="x", label="inh. neuron")
plt.legend(loc="upper right")
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
# ax.set_title("Layout of exc. (red dots) and \ninh. (blue crosses) neurons")
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_xlim(-0.5, side_length-0.5)
ax.set_xticks(np.arange(0, side_length), ["...", *np.arange(20, 20 + side_length-2), "..."])
ax.set_ylim(-0.5, side_length-0.5)
ax.set_yticks(np.arange(0, side_length), ["...", *np.arange(20, 20 + side_length-2), "..."])
# plt.tight_layout()
plt.show()

filename = os.path.join(directory, name)
fig.savefig(filename + ".svg", transparent=True)
fig.savefig(filename + ".png", transparent=True)
