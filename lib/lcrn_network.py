# -*- coding: utf-8 -*-
#
# lcrn_network.py
#
# Copyright 2017 Arvind Kumar, Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib.pyplot as plt

from lib import functimer

from lib.move import _shift_x, _shift_y, get_shift

__all__ = [
    'lcrn_gauss_targets',
]

def independent_targets(s_id, srow, trow, ncon, con_std, selfconnection=True):
    """Same look as lcrn_gauss_targets!"""
    t_pop = np.power(trow, 2)
    target_choices = np.arange(t_pop)
    if not selfconnection:
        target_choices = np.delete(target_choices, s_id)

    targets = np.random.choice(target_choices, size=ncon)
    return targets



def lcrn_gauss_targets(s_id, source_rows, target_rows, ncon, con_std, selfconnection=True, direction:int=None, shift:float=0.):
    # Margin for deleting self-connections after the targets are drawn and shifted
    tmp_ncon = int(ncon * 1.5) if direction is not None else ncon
    tmp_ncon = int(ncon * 1.5) if not selfconnection else tmp_ncon
    position = id_to_position(s_id, source_rows)
    adjusted_position, grid_scale = position_to_grid(position, source_rows, target_rows)

    targets = get_off_grid_target_positions(adjusted_position, con_std * grid_scale, tmp_ncon, selfconnection)
    targets = shift_targets(targets, direction, shift)
    target_ids = targets_to_grid(targets, target_rows)

    if not selfconnection or direction is not None:
        target_ids = target_ids[target_ids != s_id]

    return target_ids[:ncon]


def get_off_grid_target_positions(position:np.ndarray, std:float, no_of_connection:int, selfconnection:bool):
    # TODO: self connection is not used here at all!!!
    # Finds the x and y positions of the targets.
    targets = np.random.normal(scale=std, size=(2, no_of_connection))
    targets += np.asarray(position)[:, np.newaxis]
    return targets

    phi = np.random.uniform(low=-np.pi, high=np.pi, size=no_of_connection)
    radius = np.random.normal(size=no_of_connection, scale=std)
    # phi = rng.uniform(low=-np.pi, high=np.pi, size=no_of_connection)
    # radius = rng.normal(size=no_of_connection, scale=std)

    target_x = radius * np.cos(phi) + position[0]
    target_y = radius * np.sin(phi) + position[1]
    return np.asarray((target_x, target_y))
    # return np.asarray((target_x, target_y)), radius, phi


def shift_targets(targets, direction, shift):
    return (targets.T + get_shift(direction) * shift).T


def targets_to_grid(targets, target_rows):
    population_size = target_rows**2

    target_row_id = np.remainder(np.round(targets[1]) * target_rows, population_size)
    target_col_id = np.remainder(np.round(targets[0]), target_rows)
    target_ids = np.remainder(target_row_id + target_col_id, population_size)
    target_ids = target_ids.astype(int)
    return target_ids



def id_to_position(idx, source_rows)->np.ndarray:
    column = np.remainder(idx, source_rows)
    row = int(idx) // int(source_rows)
    return np.asarray((column, row))


def position_to_grid(position, source_rows, target_rows):
    """
    Scaling the grid makes it possible to adjust the connection properly.
    A grid > 1 means that a smaller population projects to a larger one.
    And vice versa.

    """
    grid_scale = target_rows / source_rows
    adjusted_position = position * grid_scale

    adjusted_position = move_to_equidistance(adjusted_position, grid_scale)
    return adjusted_position, grid_scale


def move_to_equidistance(position, grid_scale):
    if grid_scale > 1:
        position += .5
    elif grid_scale < 1:
        position -= .25
    return position
