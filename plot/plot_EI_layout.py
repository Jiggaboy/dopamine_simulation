#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:42:50 2021

@author: hauke

"""

import numpy as np
import matplotlib.pyplot as plt

side = 6
eside = np.arange(side)
x_e, y_e = np.meshgrid(eside, eside)


iside = np.arange(0.5, side, 2)
x_i, y_i = np.meshgrid(iside, iside)

plt.figure("EI_layout", figsize=(3, 3))
plt.scatter(x_e, y_e, color="r", label="excitatory neuron")
plt.scatter(x_i, y_i, color="b", marker="x", label="inhibitory neuron")
# plt.legend(loc="upper left")
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.title("Layout of exc.(red dots) and \ninh.(blue crosses) neurons")
