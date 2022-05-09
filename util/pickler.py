#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 17:07:04 2021

@author: hauke
"""

import pickle
import os
from pathlib import Path

import configuration as CF

FN_RATE = "rate.bn"
AVG_TAG = "avg_"


def save(filename: str, obj: object):
    if obj is None:
        return
    filename = prepend_dir(filename)
    ## Create dir if it does not exist
    path = Path(filename)
    os.makedirs(path.parent.absolute(), exist_ok=True)

    with open(filename, "w+b") as f:
        pickle.dump([obj], f, protocol=-1)


def load(filename: str) -> object:
    filename = prepend_dir(filename)
    with open(filename, "rb") as f:
        return pickle.load(f)[0]


def prepend_dir(filename: str, directory: str = "data"):
    return os.path.join(directory, filename)


def get_filename(postfix: str = None):
    if postfix:
        fname = FN_RATE.replace(".", f"_{postfix}.")
    else:
        fname = FN_RATE
    return fname


def save_rate(obj: object, postfix: str = None, sub_directory:str=None) -> None:
    fname = get_filename(postfix)
    if sub_directory:
        fname = prepend_dir(fname, sub_directory)
    print(f"Save rates to {fname}!")
    save(fname, obj)


def load_rate(postfix:str=None, skip_warmup:bool=False, exc_only:bool=False, sub_directory:str=None, config=None) -> object:
    fname = get_filename(postfix)
    if sub_directory:
        fname = prepend_dir(fname, sub_directory)

    rate = load(fname)
    if skip_warmup:
        # rate = rate[:, CF.WARMUP:]
        rate = rate[:, config.WARMUP:]
    if exc_only:
        # rate = rate[:CF.NE]
        rate = rate[:int(config.rows**2)]
    return rate



def save_avg_rate(avgRate, postfix, sub_directory:str, **kwargs):
    save_rate(avgRate, AVG_TAG + postfix, sub_directory)


def load_average_rate(postfix, **kwargs):
    return load_rate(AVG_TAG + postfix, **kwargs)
