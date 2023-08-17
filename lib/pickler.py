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
import cflogger
logger = cflogger.getLogger()

import os
import pickle
import numpy as np
from pathlib import Path

from lib import SequenceCounter, functimer
import lib.universal as UNI


FN_RATE = "rate.bn"
AVG_TAG = "avg_"
SEQ_TAG = "seq_"
SEQ_DB_TAG = "seq_db_"
SEQ_CLUSTER_DB_TAG = "seq_db_cluster_"
PCA_TAG = "pca_"
ANGLE_DUMPER = "angle_dumper_"
SPIKE_TRAIN = "spike_train_"

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



def save_sequence(sequence, postfix:str, sub_directory:str, **kwargs):
    save(SEQ_TAG + postfix, sequence, sub_directory)


def load_sequence(postfix, **kwargs):
    return load(SEQ_TAG + postfix, **kwargs)



def save_pca(pca, postfix:str, **kwargs):
    save(PCA_TAG + postfix, pca, **kwargs)

def load_pca(postfix, **kwargs):
    return load(PCA_TAG + postfix, **kwargs)



def save_angle_dumper(angle_dumper, **kwargs):
    save(ANGLE_DUMPER + angle_dumper.tag, angle_dumper, **kwargs)

def load_angle_dumper(tag:str, **kwargs):
    return load(ANGLE_DUMPER + tag, **kwargs)



def save_db_sequence(sequence, postfix:str, sub_directory:str, **kwargs)->None:
    save(SEQ_DB_TAG + postfix, sequence, sub_directory)


def load_db_sequence(postfix, **kwargs)->object:
    return load(SEQ_DB_TAG + postfix, **kwargs)



def save_db_cluster_sequence(sequence, postfix:str, sub_directory:str, **kwargs)->None:
    save(SEQ_CLUSTER_DB_TAG + postfix, sequence, sub_directory)


def load_db_cluster_sequence(postfix, **kwargs)->object:
    return load(SEQ_CLUSTER_DB_TAG + postfix, **kwargs)



def save_spike_train(spike_train:object, postfix:str, sub_directory:str, **kwargs)->None:
    save(SPIKE_TRAIN + postfix, spike_train, sub_directory)


def load_spike_train(postfix, **kwargs)->object:
    return load(SPIKE_TRAIN + postfix, **kwargs)


def create_dir(filename:str):
    """Creates directories such that the filename is valid."""
    path = Path(filename)
    os.makedirs(path.parent.absolute(), exist_ok=True)


def load_coordinates_and_rate(cfg:object, tag:str):
    """
    Loads the coordinates and the rates (according to the full tag including the details) of the exc. populattion.
    """
    coordinates = UNI.get_coordinates(nrows=cfg.rows, step=1)
    rate = load_rate(tag, sub_directory=cfg.sub_dir, config=cfg, skip_warmup=True, exc_only=True)
    return coordinates, rate
