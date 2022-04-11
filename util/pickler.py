#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 17:07:04 2021

@author: hauke
"""

import pickle
import os

import configuration as CF

FN_RATE = "rate.bn"


def save(filename: str, obj: object):
    if obj is None:
        return
    filename = prepend_dir(filename)
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


def save_rate(obj: object, postfix: str = None) -> None:
    fname = get_filename(postfix)
    print(f"Save rates to {fname}!")
    save(fname, obj)


def load_rate(postfix: str = None, skip_warmup: bool = False, exc_only: bool = False) -> object:
    fname = get_filename(postfix)

    rate = load(fname)
    if skip_warmup:
        rate = rate[:, CF.WARMUP:]
    if exc_only:
        rate = rate[:CF.NE]
    return rate
