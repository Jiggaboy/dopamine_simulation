#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:59:39 2021

@author: hauke
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

import cflogger
log = cflogger.getLogger()

from custom_class.population import Population
from util.pickler import prepend_dir


def set_seed(use_constant_seed: bool=None):
    print(f"Use constant seed: {bool(use_constant_seed)}")
    if use_constant_seed:
        np.random.seed(0)
    else:
        np.random.seed(None)


def get_tag_ident(*tags, delimiter:str="_"):
    """Assembles an identifier placing the delimiter between the tags."""
    return delimiter.join((str(t) for t in tags))


def get_fig_filename(tag:str, format_="png"):
    fname = prepend_dir(tag, directory="figures")
    fname += "." + format_
    return fname



def find_tags(config, t:tuple)->list:
    """
    Finds all the tags in the config starting with elements in t.
    """
    tags = []
    for tag_name in t:
        tags.extend([t for t in config.get_all_tags() if t.startswith(tag_name)])
    return tags


def get_parameter_space():
    p_space = OrderedDict({
        "center": tuple(CF.center_range.keys()),
        "radius": CF.RADIUSES,
        "n_neurons": CF.AMOUNT_NEURONS,
        "m_synapses": CF.P_synapses,
        "p_weight": np.asarray(100*np.asarray(CF.PERCENTAGES), dtype=int),
    })

    space = np.meshgrid(*p_space.values(), indexing="ij")
    grid = np.asarray(list(zip(*(s.ravel() for s in space))))
    return p_space.keys(), grid


def sort_space(space, centername:str, order:tuple=None):
    """rOder may contain 'r', 'm', 'n', and/or 'p'. Default order is: ('r', 'n', 'm', 'p')"""
    order = order or ["r", "n", "m", "p"]
    space = space[space[:, 0] == centername]
    df_space = pd.DataFrame(space[:, 1:], columns=("r", "n", "m", "p"), dtype=float)

    return df_space.sort_values(order).to_numpy()


def idx2patch(idx):
    """Takes a neuron IDs as input and return a array of exc. neurons """
    null_space = np.zeros(CF.NE, dtype=int)
    null_space[idx] = 1
    return null_space


def patch2idx(patch):
    """Takes a patch and returns the IDs of neurons."""
    return patch.nonzero()[0]


def ensure_valid_operation_range(r_dot:np.ndarray, minmax:float=2000.)->np.ndarray:
    r_dot[r_dot > minmax] = minmax
    r_dot[r_dot < -minmax] = -minmax
    return r_dot
