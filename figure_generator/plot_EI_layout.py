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
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
#===============================================================================
# CONSTANTS
#===============================================================================

name = "EI_layout"
directory = "figures"
side_length = 6
pos = np.arange(side_length)
x_pos_exc, y_pos_exc = np.meshgrid(pos, pos)


iside = np.arange(0.5, side_length, 2)
x_pos_inh, y_pos_inh = np.meshgrid(iside, iside)

fig = plt.figure(name, figsize=(3, 3))
plt.scatter(x_pos_exc, y_pos_exc, color="r", label="excitatory neuron")
plt.scatter(x_pos_inh, y_pos_inh, color="b", marker="x", label="inhibitory neuron")
# plt.legend(loc="upper left")
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.title("Layout of exc.(red dots) and \ninh.(blue crosses) neurons")
plt.tight_layout()
plt.show()

filename = os.path.join(directory, name)
fig.savefig(filename + ".svg")
fig.savefig(filename + ".png")
