#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""
from cflogger import logger

import numpy as np
import os
from collections.abc import Iterable

import lib.dopamine as DOP


## Constants
TAG_DELIMITER = "_"
TAG_NAME_INDEX = 0
TAG_RADIUS_INDEX = 1
TAG_SEED_INDEX = -1


def log_status(cfg:object, radius, name, amount, percent):
    logger.info("Simulation" \
          + f" radius: {cfg.RADIUSES.index(radius) + 1}/{len(cfg.RADIUSES)};"
          + f" name: {name};"
          + f" amount: {cfg.AMOUNT_NEURONS.index(amount) + 1}/{len(cfg.AMOUNT_NEURONS)};"
          + f" percent: {cfg.PERCENTAGES.index(percent) + 1}/{len(cfg.PERCENTAGES)};")


def get_tag_ident(*tags, delimiter:str=TAG_DELIMITER):
    """Assembles an identifier placing the delimiter between the tags."""
    return delimiter.join((str(t) for t in tags))


def play_beep(repeat:int=3, pause:float=0.2):
    beep = lambda x: os.system(f"echo -n '\a'; sleep {pause};" * x)
    beep(repeat)


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


def patch2idx(patch):
    """Takes a patch and returns the IDs of neurons."""
    return patch.nonzero()[0]


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


def neurons_from_center(center:list, radius:float, nrows:int)->list:
    patches = [DOP.circular_patch(nrows, c, radius) for c in center]
    neurons = [patch2idx(patch) for patch in patches]
    return neurons


def yes_no(question:str, answer:bool=None) -> bool:
    if answer is not None:
        return answer
    answer = input(question + " (y/n)")
    return answer.lower().strip() == "y"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
