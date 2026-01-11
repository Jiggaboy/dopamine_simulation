#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1b'
#===============================================================================
# HISTORY
#===============================================================================
# Version 0.1b:
#     - Remove unused find_tags, get_center_from_list functions.


#===============================================================================
# IMPORTS
#===============================================================================

from cflogger import logger

import numpy as np
import os
from pathlib import Path, PosixPath
from constants import DATA_DIR, FIGURE_DIR
from collections.abc import Iterable

# import lib.dopamine as DOP


## Constants
TAG_DELIMITER = "_"
TAG_NAME_INDEX = 0
TAG_RADIUS_INDEX = 1
TAG_PERCENTAGE_INDEX = 3
TAG_SEED_INDEX = -1




def get_neurons_from_patch(area:np.ndarray, amount:int, repeat_samples:bool=None) -> np.ndarray:
    if repeat_samples is None:
        generator = np.random.default_rng()
        return generator.choice(area.nonzero()[0], amount, replace=False)
    elif repeat_samples:
        logger.info(f"Set seed to {repeat_samples}.")
        np.random.seed(repeat_samples)
        return np.random.choice(area.nonzero()[0], amount, replace=False)
    else:
        logger.info("Set seed to 0 (default).")
        np.random.seed(0)
        return np.random.choice(area.nonzero()[0], amount, replace=False)




def log_status(cfg:object, radius, name, amount, percent):
    logger.info("Simulation" \
          + f" radius: {cfg.radius.index(radius) + 1}/{len(cfg.radius)};"
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


def split_percentage_from_tag(tag:str)->tuple:
    return tag.split(TAG_DELIMITER)[TAG_PERCENTAGE_INDEX]


def name_from_tag(tag:str)->tuple:
    return tag.split(TAG_DELIMITER)[TAG_NAME_INDEX]


def radius_from_tag(tag:str)->tuple:
    return tag.split(TAG_DELIMITER)[TAG_RADIUS_INDEX]


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


def squeeze_labels(labels:np.ndarray):
    """Squeezes labels to integers between 0 and len(set(labels))."""
    unique_labels = set(labels)
    for i, label in enumerate(sorted(unique_labels)):
        labels[labels == label] = i
    return labels


def yes_no(question:str, answer:bool=None) -> bool:
    if answer is not None:
        return answer
    answer = input(question + " (y/n)")
    return answer.lower().strip() == "y"


def prepend_dir(filename: str, directory: str = DATA_DIR) -> PosixPath:
    # Update to pathlib in v0.1a
    return Path(directory).joinpath(filename)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
