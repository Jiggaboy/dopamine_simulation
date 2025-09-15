#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2c'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import os
import pickle
import numpy as np
from pathlib import Path
import pandas as pd


import lib.universal as UNI

from constants import DATA_DIR, FIGURE_DIR
from constants import TAG_SYNAPTIC_INPUT

FN_RATE = "rate.bn"
AVG_TAG = "avg_"
SPIKE_TRAIN = "spike_train_"
SEQ_CROSS_CENTER = "seq_cross_center_"

ANIMATION_SUFFIX = ".gif"
FIGURE_SUFFIX = ".svg"
FIGURE_ALTERNATIVE_SUFFIX = ".png"


#===============================================================================
# FIGURES
#===============================================================================

def save_figure(filename:str, figure:object, sub_directory:str=None, **kwargs):
    """
    Saves the figure in the subdirectory of the config.
    """
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename, FIGURE_DIR)
    create_dir(filename)

    figure.savefig(filename + FIGURE_SUFFIX, **kwargs)
    figure.savefig(filename + FIGURE_ALTERNATIVE_SUFFIX, **kwargs)


#===============================================================================
# ANIMATION
#===============================================================================

def save_animation(filename:str, animation:object, sub_directory:str=None):
    """
    Saves the animation in the subdirectory of the config.
    """
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename, FIGURE_DIR)
    create_dir(filename)
    animation.save(filename + ANIMATION_SUFFIX)



#===============================================================================
# FURTHER METHODS
#===============================================================================


def save(filename: str, obj: object, sub_directory:str=None, mode:str="pic"):
    if obj is None:
        logger.error("No object given. Save cancelled...")
        return
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename)

    create_dir(filename)

    if mode == "pic":
        with open(filename, "w+b") as f:
            pickle.dump([obj], f, protocol=-1)
    elif mode == "pd":
        obj.to_pickle(filename)
    else:
        raise ValueError("Save: No valid mode given!")


def load(filename:str, sub_directory:str=None, mode:str="pic") -> object:
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename)

    if mode == "pic":
        with open(filename, "rb") as f:
            return pickle.load(f)[0]
    elif mode == "pd":
        return pd.read_pickle(filename)
    else:
        raise ValueError("Save: No valid mode given!")


def path_exists(path:str, skip_datadir:bool=False):
    """
    Check whether a path exists in the data directory.
    In base directory if {skip_datadir} is True.
    """
    if not skip_datadir:
        path = prepend_dir(path)
    return Path(path).exists()

def prepend_dir(filename:str, directory:str = DATA_DIR):
    return os.path.join(directory, filename)


def get_filename(postfix: str = None):
    if postfix:
        fname = FN_RATE.replace(".", f"_{postfix}.")
    else:
        fname = FN_RATE
    return fname


#===============================================================================
# CONNECTIVITY MATRIX
#===============================================================================


def save_conn_matrix(filename:str, conn_matrix:object, EE_only:bool=False, sub_directory:str=None)->None:
    if conn_matrix is None:
        logger.error("No object given. Save cancelled...")
        return
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename)

    create_dir(filename)

    arr = conn_matrix._EE if EE_only else conn_matrix.connections
    np.savez_compressed(filename, W=arr)


#===============================================================================
# (AVERAGE) RATES
#===============================================================================

def save_rate(obj: object, postfix: str = None, sub_directory:str=None) -> None:
    fname = get_filename(postfix)
    if sub_directory:
        fname = prepend_dir(fname, sub_directory)
    logger.info(f"Save rates to {fname}!")
    save(fname, obj)


def load_rate(postfix:str=None, skip_warmup:bool=False, exc_only:bool=False, sub_directory:str=None, config=None, no_return:bool=False)->object:
    fname = get_filename(postfix)
    if sub_directory:
        fname = prepend_dir(fname, sub_directory)


    rate = load(fname)
    if skip_warmup:
        rate = rate[:, -int(config.sim_time):]
    if exc_only:
        rate = rate[:int(config.rows**2)]
    return rate



def save_avg_rate(avgRate, postfix:str, sub_directory:str, **kwargs):
    save_rate(avgRate, AVG_TAG + postfix, sub_directory)


def load_average_rate(postfix:str, dry_run:bool=False, **kwargs):
    if dry_run:
        return AVG_TAG + postfix
    try:
        return load_rate(AVG_TAG + postfix, **kwargs)
    except FileNotFoundError:
        return None

#===============================================================================
# SYNAPTIC INPUT
#===============================================================================

def save_synaptic_input(obj: object, postfix: str = None, sub_directory:str=None) -> None:
    fname = TAG_SYNAPTIC_INPUT.format(postfix)
    if sub_directory:
        fname = prepend_dir(fname, sub_directory)
    logger.info(f"Save rates to {fname}!")
    save(fname, obj)

def load_synaptic_input(postfix: str = None, sub_directory:str=None) -> None:
    fname = TAG_SYNAPTIC_INPUT.format(postfix)
    if sub_directory:
        fname = prepend_dir(fname, sub_directory)
    return load(fname)


#===============================================================================
# FURTHER METHODS
#===============================================================================

def _save_spike_train(spike_train:object, postfix:str, sub_directory:str, **kwargs)->None:
    save(SPIKE_TRAIN + postfix, spike_train, sub_directory)


def _load_spike_train(postfix, sub_directory:str, **kwargs)->object:
    return load(SPIKE_TRAIN + postfix, sub_directory, **kwargs)


def save_spike_train(tag:str, config:object, data:np.ndarray, labels:np.ndarray, eps:float=None, min_samples:float=None, return_identifier:bool=False) -> None:
    eps = eps if eps is not None else config.analysis.sequence.eps
    min_samples = min_samples if min_samples is not None else config.analysis.sequence.min_samples

    identifier, filename = get_spike_train_identifier_filename(tag, eps, min_samples)
    identifier["data"] = data
    identifier["labels"] = labels
    _save_spike_train(identifier, filename, sub_directory=config.sub_dir)

from functools import cache
@cache
def load_spike_train(tag:str, config:object, eps:float=None, min_samples:float=None, return_identifier:bool=False) -> (np.ndarray, np.ndarray):
    eps = eps if eps is not None else config.analysis.sequence.eps
    min_samples = min_samples if min_samples is not None else config.analysis.sequence.min_samples

    identifier, filename = get_spike_train_identifier_filename(tag, eps, min_samples)
    obj = _load_spike_train(filename, sub_directory=config.sub_dir)
    # logger.info(f"Load spike train of tag: {tag}")
    if return_identifier:
        return return_identifier
    return obj["data"], obj["labels"]


def get_spike_train_identifier_filename(tag, eps, min_samples):
    identifier = {
        "tag": tag,
        "eps": str(eps),
        "min_samples": str(min_samples),
    }
    filename = "_".join(identifier.values())
    return identifier, filename


def create_dir(filename:str):
    """Creates directories such that the filename is valid."""
    path = Path(filename)
    os.makedirs(path.parent.absolute(), exist_ok=True)


def datafile_exists(tag:str, sub_directory:str=None, **kwargs) -> bool:
    filename = get_filename(tag)
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename)
    return Path(filename).is_file()


def load_coordinates_and_rate(config:object, tag:str):
    """
    Loads the coordinates and the rates (according to the full tag including the details) of the exc. populattion.
    """
    coordinates = UNI.get_coordinates(nrows=config.rows, step=1)
    rate = load_rate(tag, sub_directory=config.sub_dir, config=config, skip_warmup=True, exc_only=True)
    return coordinates, rate


def load_sequence_at_center(tag:str, center:tuple, config:object) -> object:
    filename = _get_filename_sequence_at_center(tag, center, config)
    try:
        return load(filename, sub_directory=config.sub_dir, mode="pd")
    except FileNotFoundError:
        return None


def save_sequence_at_center(sequence_at_center:np.ndarray, tag:str, center:tuple, config:object):
    filename = _get_filename_sequence_at_center(tag, center, config)
    save(filename, sequence_at_center, sub_directory=config.sub_dir, mode="pd")


def _get_filename_sequence_at_center(tag:str, center:tuple, config:object) -> object:
    separator = "_"
    _, name = get_spike_train_identifier_filename(tag, config.analysis.sequence.eps, config.analysis.sequence.min_samples)
    return SEQ_CROSS_CENTER + name + separator + separator.join(map(str, center))
