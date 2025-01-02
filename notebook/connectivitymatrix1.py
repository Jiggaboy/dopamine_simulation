#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Runs the model with different kinds of configurations.

Description:


A sample script can be found in "figure_generator/in_out_degree.py"

"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
# Copyright 2017 Sebastian Spreizer
# The MIT License
__author__ = ['Sebastian Spreizer', 'Hauke Wernecke']
__contact__ = 'hower@kth.se'
__licence__ = 'MIT License'
__version__ = '0.1'


#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
import numpy as np

import lcrn_network as lcrn


# Landscape - Possible values:
#     - random (Random preferred direction, seed:int)
#     - independent (Does not have a preferred direction nor a spatial distribution, seed:int)
#     - symmetric (No pref. direction, seed:int)
#     - Perlin (Pref. direction according to the Perlin noise) {Parameter: size:int, base:int, seed:int}
#     - Perlin_uniform (as for Perlin but euqally distributed throughout the 8 directions)
#     - homogeneous (all have the same preferred direction) {Takes a preferred direction 'phi' with values between 0-7, seed:int}


from dataclasses import dataclass

@dataclass
class Group:
    rows: int
    std: float

    @property
    def quantity(self):
        return self.rows ** 2

@dataclass
class Connection:
    idx: int
    source: Group
    target: Group
    connection_probability: float
    allow_self_connections: bool = None

    @property
    def no_of_targets(self)->int:
        return int(self.connection_probability * self.target.quantity)

    @property
    def std(self)->float:
        return self.source.std


    def get_all(self):
        return self.idx, self.source.rows, self.target.rows, self.no_of_targets, self.std, self.allow_self_connections


def EI_networks(nrowE, stdE:float, stdI:float, connection_probability:float, shift_matrix:np.ndarray=None, shift:float=1., **kwargs):
    grp_E = Group(nrowE, stdE)
    grp_I = Group(nrowE // 2, stdI)

    EE_setup = grp_E, grp_E, False, not shift_matrix is None
    EI_setup = grp_E, grp_I, True
    IE_setup = grp_I, grp_E, True
    II_setup = grp_I, grp_I, False
    setups = (EE_setup, EI_setup, IE_setup, II_setup)

    fct_find_targets = lcrn.independent_targets if shift_matrix is None else lcrn.lcrn_gauss_targets

    def find_neurons(source, target, allow_self_connections, shift=None):
        # self connections within EE/II are removed anyways. it only is taken into account for finding targets before applying the shift.
        conmat = np.zeros((source.quantity, target.quantity))
        for idx in range(source.quantity):
            cnn = Connection(idx, source, target, connection_probability, allow_self_connections=allow_self_connections)
            direction = None if shift is None else shift_matrix[idx]
            postsynaptic_ids = fct_find_targets(*cnn.get_all(), direction=direction, shift=shift)

            conmat[idx] = id_to_neuron(postsynaptic_ids, target)
        return conmat


    conmats = []
    for s in setups:
        conmat = find_neurons(*s)
        conmats.append(conmat)
        if kwargs.get("EE_only", False):
            break
    return conmats


def id_to_neuron(target_id, target_grp):
    return np.histogram(target_id, bins=range(target_grp.quantity + 1))[0]

def as_full_connectivity_matrix(EE, IE, EI, II):
    """Full adjacency matrix of exc. and inh. population."""
    E_EI = np.concatenate((EE, IE), axis=1)
    I_EI = np.concatenate((EI, II), axis=1)
    W = np.concatenate((E_EI, I_EI)).T
    return W
