# -*- coding: utf-8 -*-
#
# lcrn_network.py
#
# Copyright 2017 Arvind Kumar, Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'lcrn_gauss_targets',
]


def lcrn_gauss_targets(s_id, srow, trow, ncon, con_std, selfconnection=True):
    grid_scale = trow / srow
    s_x = np.remainder(s_id, srow)  # column id
    s_y = int(s_id) // int(srow)  # row id
    s_x1 = s_x * grid_scale  # column id in the new grid
    s_y1 = s_y * grid_scale  # row_id in the new grid
    if grid_scale > 1:
        s_x1 += .5
        s_y1 += .5
    if grid_scale < 1:
        s_x1 -= .25
        s_y1 -= .25
    con_std *= grid_scale
    # pick up ncol values for phi and radius
    phi = np.random.uniform(low=-np.pi, high=np.pi, size=ncon)
    radius = con_std * np.random.randn(ncon)
    if not selfconnection:
        radius[radius>0] = radius[radius>0] + .5
        radius[radius<0] = radius[radius<0] - .5
    # if s_id==0:
    #     plt.figure()
    #     plt.hist(radius, 20, (-20, 21))

    t_x = radius * np.cos(phi) + s_x1
    t_y = radius * np.sin(phi) + s_y1
    # target_ids = np.remainder(np.round(t_y) * trow + np.round(t_x), trow * trow)
    N = trow * trow
    target_row = np.remainder(floor_towards_zero(t_y) * trow, N)
    target_col = np.remainder(floor_towards_zero(t_x), trow)
    target_ids = np.remainder(target_row + target_col, N)
    target = target_ids.astype(int)
    # delays = np.abs(radius) / trow
    delays = None
    return target, delays


def floor_towards_zero(number:(int, np.ndarray)):
    return np.round(number)
    # return np.floor(number)
    # return np.sign(number) * np.floor(np.abs(number))
