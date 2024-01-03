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
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import os
import pickle
import numpy as np
from pathlib import Path

import lib.universal as UNI


FN_RATE = "rate.bn"
AVG_TAG = "avg_"
SEQ_CLUSTER_DB_TAG = "seq_db_cluster_"
SPIKE_TRAIN = "spike_train_"
SEQ_CROSS_CENTER = "seq_cross_center_"

DATA_DIR = "data"
FIGURE_DIR = "figures"

ANIMATION_SUFFIX = ".gif"
FIGURE_SUFFIX = ".svg"
FIGURE_ALTERNATIVE_SUFFIX = ".png"



def get_fig_filename(tag:str, format_="png"):
    fname = prepend_dir(tag, directory="figures")
    fname += "." + format_
    return fname


def save_animation(filename:str, animation:object, sub_directory:str):
    """
    Saves the animation in the subdirectory of the config.
    """
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename, FIGURE_DIR)
    create_dir(filename)

    filename += ANIMATION_SUFFIX
    animation.save(filename)


def save_figure(filename:str, figure:object, sub_directory:str=None):
    """
    Saves the figure in the subdirectory of the config.
    """
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename, FIGURE_DIR)
    create_dir(filename)

    figure.savefig(filename + FIGURE_SUFFIX)
    figure.savefig(filename + FIGURE_ALTERNATIVE_SUFFIX)


def save(filename: str, obj: object, sub_directory:str=None):
    if obj is None:
        logger.error("No object given. Save cancelled...")
        return
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename)

    create_dir(filename)

    with open(filename, "w+b") as f:
        pickle.dump([obj], f, protocol=-1)


def load(filename: str, sub_directory:str=None) -> object:
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename)

    with open(filename, "rb") as f:
        return pickle.load(f)[0]


def prepend_dir(filename: str, directory: str = DATA_DIR):
    return os.path.join(directory, filename)


def get_filename(postfix: str = None):
    if postfix:
        fname = FN_RATE.replace(".", f"_{postfix}.")
    else:
        fname = FN_RATE
    return fname



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


def save_rate(obj: object, postfix: str = None, sub_directory:str=None) -> None:
    fname = get_filename(postfix)
    if sub_directory:
        fname = prepend_dir(fname, sub_directory)
    logger.info(f"Save rates to {fname}!")
    save(fname, obj)


def load_rate(postfix:str=None, skip_warmup:bool=False, exc_only:bool=False, sub_directory:str=None, config=None)->object:
    fname = get_filename(postfix)
    if sub_directory:
        fname = prepend_dir(fname, sub_directory)

    rate = load(fname)
    if skip_warmup:
        rate = rate[:, -int(config.sim_time):]
    if exc_only:
        rate = rate[:int(config.rows**2)]
    return rate



def save_avg_rate(avgRate, postfix, sub_directory:str, **kwargs):
    save_rate(avgRate, AVG_TAG + postfix, sub_directory)


def load_average_rate(postfix, **kwargs):
    return load_rate(AVG_TAG + postfix, **kwargs)


def save_db_cluster_sequence(sequence, postfix:str, sub_directory:str, **kwargs)->None:
    save(SEQ_CLUSTER_DB_TAG + postfix, sequence, sub_directory)


def load_db_cluster_sequence(postfix, **kwargs)->object:
    return load(SEQ_CLUSTER_DB_TAG + postfix, **kwargs)



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


def load_spike_train(tag:str, config:object, eps:float=None, min_samples:float=None, return_identifier:bool=False) -> (np.ndarray, np.ndarray):
    eps = eps if eps is not None else config.analysis.sequence.eps
    min_samples = min_samples if min_samples is not None else config.analysis.sequence.min_samples

    identifier, filename = get_spike_train_identifier_filename(tag, eps, min_samples)
    obj = _load_spike_train(filename, sub_directory=config.sub_dir)
    logger.info(f"Load spike train of tag: {tag}")
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


def load_coordinates_and_rate(config:object, tag:str):
    """
    Loads the coordinates and the rates (according to the full tag including the details) of the exc. populattion.
    """
    coordinates = UNI.get_coordinates(nrows=config.rows, step=1)
    rate = load_rate(tag, sub_directory=config.sub_dir, config=config, skip_warmup=True, exc_only=True)
    return coordinates, rate


def load_sequence_at_center(tag:str, center:tuple, config:object) -> object:
    filename = _get_filename_sequence_at_center(tag, center)
    try:
        return load(filename, sub_directory=config.sub_dir)
    except FileNotFoundError:
        return None


def save_sequence_at_center(sequence_at_center:np.ndarray, tag:str, center:tuple, config:object):
    filename = _get_filename_sequence_at_center(tag, center)
    save(filename, sequence_at_center, sub_directory=config.sub_dir)


def _get_filename_sequence_at_center(tag:str, center:tuple) -> object:
    separator = "_"
    return SEQ_CROSS_CENTER + tag + separator + separator.join(map(str, center))
