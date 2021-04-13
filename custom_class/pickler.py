#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 17:07:04 2021

@author: hauke
"""

import pickle

import configuration as CF

FN_RATE = "rate.bn"


def save(filename:str, obj:object):
    with open(filename, "wb") as f:
        pickle.dump([obj], f, protocol=-1)

def load(filename:str)->object:
    with open(filename, "rb") as f:
        return pickle.load(f)[0]


def save_rate(obj:object)->None:
    save(FN_RATE, obj)


def load_rate(postfix:str=None, skip_warmup:bool=False, exc_only:bool=False)->object:
    if postfix:
        fname = FN_RATE.replace(".", f"_{postfix}.")
    else:
        fname = FN_RATE

    rate = load(fname)
    if skip_warmup:
        rate = rate[:, CF.WARMUP:]
    if exc_only:
        rate = rate[:CF.NE]
    return rate
