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
__version__ = '0.2'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

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

ax.scatter(x_pos_exc, y_pos_exc, color="r", label="excitatory neuron")
ax.scatter(x_pos_inh, y_pos_inh, color="b", marker="x", label="inhibitory neuron")
# plt.legend(loc="upper left")
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
# ax.set_title("Layout of exc. (red dots) and \ninh. (blue crosses) neurons")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xticks([0, 2, 4])
ax.set_yticks([0, 2, 4])
plt.tight_layout()
plt.show()

filename = os.path.join(directory, name)
fig.savefig(filename + ".svg", transparent=True)
fig.savefig(filename + ".png", transparent=True)
