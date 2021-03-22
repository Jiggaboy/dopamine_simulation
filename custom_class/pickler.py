#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 17:07:04 2021

@author: hauke
"""

import pickle

FN_RATE = "rate.bn"


def save(filename:str, obj:object):
    with open(filename, "wb") as f:
        pickle.dump([obj], f, protocol=-1)

def load(filename:str)->object:
    with open(filename, "rb") as f:
        return pickle.load(f)[0]


def save_rate(obj:object)->None:
    save(FN_RATE, obj)


def load_rate()->object:
    return load(FN_RATE)
