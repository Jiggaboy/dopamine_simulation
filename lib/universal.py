#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:59:39 2021

@author: hauke
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from collections.abc import Iterable

import cflogger
log = cflogger.getLogger()

## Constants
TAG_DELIMITER = "_"
TAG_NAME_INDEX = 0
TAG_RADIUS_INDEX = 1
TAG_SEED_INDEX = -1


def log_status(cfg:object, radius, name, amount, percent):
    log.info("Simulation" \
          + f" radius: {cfg.RADIUSES.index(radius) + 1}/{len(cfg.RADIUSES)};"
          + f" name: {name};"
          + f" amount: {cfg.AMOUNT_NEURONS.index(amount) + 1}/{len(cfg.AMOUNT_NEURONS)};"
          + f" percent: {cfg.PERCENTAGES.index(percent) + 1}/{len(cfg.PERCENTAGES)};")


def get_tag_ident(*tags, delimiter:str=TAG_DELIMITER):
    """Assembles an identifier placing the delimiter between the tags."""
    return delimiter.join((str(t) for t in tags))


def split_seed_from_tag(tag:str)->tuple:
    return tag.rsplit(TAG_DELIMITER, 1)


def name_from_tag(tag:str)->tuple:
    return tag.split(TAG_DELIMITER)[TAG_NAME_INDEX]


def radius_from_tag(tag:str)->tuple:
    return tag.split(TAG_DELIMITER)[TAG_RADIUS_INDEX]


def find_tags(config, t:tuple)->list:
    """
    Finds all the tags in the config starting with elements in t.
    """
    tags = []
    for tag_name in t:
        tags.extend([t for t in config.get_all_tags() if t.startswith(tag_name)])
    return tags


def sort_space(space, centername:str, order:tuple=None):
    """rOder may contain 'r', 'm', 'n', and/or 'p'. Default order is: ('r', 'n', 'm', 'p')"""
    order = order or ["r", "n", "m", "p"]
    space = space[space[:, 0] == centername]
    df_space = pd.DataFrame(space[:, 1:], columns=("r", "n", "m", "p"), dtype=float)

    return df_space.sort_values(order).to_numpy()


def patch2idx(patch):
    """Takes a patch and returns the IDs of neurons."""
    return patch.nonzero()[0]


def ensure_valid_operation_range(r_dot:np.ndarray, minmax:float=2000.)->np.ndarray:
    r_dot[r_dot > minmax] = minmax
    r_dot[r_dot < -minmax] = -minmax
    return r_dot


def append_spot(spots:list, tag:str, center:tuple):
    """ Appends the the tag and the center as a tuple to spots."""
    return spots.append((tag, center))



def binarize_rate(rate:np.ndarray, threshold:float=0.5):
    """ Every activation above threshold is turned to a 1, 0 otherwise."""
    rate[rate >= threshold] = 1
    rate[rate < threshold] = 0
    return rate


def make_iterable(element):
    if isinstance(element, str):
        return (element, )
    if not isinstance(element, Iterable):
        return (element, )
    return element


def get_center_from_list(tag_spots:list)->list:
    """
    Retrieves all center across different tags in a single list.
    """
    all_center = []
    for _, center in tag_spots:
        all_center.extend(center)
    return all_center


def get_coordinates(nrows:int, step:int=1)->np.ndarray:
    """
    Generates coordinates for a square grid with a side length of {nrows}.
    Distances between positions is determined by the parameter {step}.

    Parameters
    ----------
    nrows : int
        Side length of the grid.
    step : int, optional
        Distance between positions. The default is 1.

    Returns
    -------
    coordinates : np.ndarray
        2D-array with the x- and y-positions.

    """
    positions = np.arange(0, nrows, step)
    x, y = np.meshgrid(positions, positions)
    coordinates = np.asarray(list(zip(x.ravel(), y.ravel())))
    return coordinates


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


################################################### Depreciated?? ###################################################
def calculate_direction(x, bins=8, **kwargs):
    print("Not depreicated!!!!")
    rad = 2 * np.pi
    u = np.cos(x / bins * rad)
    v = np.sin(x / bins * rad)
    return u, v
