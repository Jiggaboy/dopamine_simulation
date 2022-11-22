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

from custom_class.population import Population
from lib.pickler import prepend_dir

## Constants
TAG_DELIMITER = "_"
TAG_NAME_INDEX = 0
TAG_RADIUS_INDEX = 1
TAG_SEED_INDEX = -1


def get_tag_ident(*tags, delimiter:str=TAG_DELIMITER):
    """Assembles an identifier placing the delimiter between the tags."""
    return delimiter.join((str(t) for t in tags))


def get_fig_filename(tag:str, format_="png"):
    fname = prepend_dir(tag, directory="figures")
    fname += "." + format_
    return fname


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
