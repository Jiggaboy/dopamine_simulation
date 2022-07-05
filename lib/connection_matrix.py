# -*- coding: utf-8 -*-
#
# connection_matrix.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib.pyplot as plt

import lib.lcrn_network as lcrn
import lib.connectivity_landscape as cl

from lib import Group, Connection



def set_seed(seed):
    if seed is None:
        print("set random seed")
        np.random.seed()
        seed = np.random.randint(1000)
    else:
        print(f"set seed: {seed}")
        np.random.seed(seed)


def get_shift_matrix(landscape:str, excitatory_group:Group)->np.ndarray:
    return cl.__dict__[landscape.mode](excitatory_group.rows, landscape.params)


def EI_networks(landscape, nrowE, **kwargs):
    grp_E = Group(nrowE, landscape.stdE)
    grp_I = Group(nrowE, landscape.stdI)

    set_seed(landscape.seed)


    shift_matrix = get_shift_matrix(landscape, grp_E)


    EE_setup = grp_E, grp_E, landscape.is_asymmetric, shift_matrix
    EI_setup = grp_E, grp_I, True
    IE_setup = grp_I, grp_E, True
    II_setup = grp_I, grp_I, False
    setups = (EE_setup, EI_setup, IE_setup, II_setup)


    fct_find_targets = lcrn.independent_targets if landscape.is_independent else lcrn.lcrn_gauss_targets
    def find_neurons(source, target, allow_self_connections, shift=None):
        # self connections within EE/II are removed anyways. it only is taken into account for finding targets before applying the shift.
        conmat = np.zeros((source.quantity, target.quantity))
        for idx in range(source.quantity):
            cnn = Connection(idx, source, target, landscape.connection_probability, allow_self_connections=allow_self_connections)

            direction = None if shift is None else shift_matrix[idx]

            postsynaptic_ids = fct_find_targets(*cnn.get_all(), direction=direction, shift=landscape.shift)

            conmat[idx] = id_to_neuron(postsynaptic_ids, target)
        return conmat


    conmats = []
    for s in setups:
        conmat = find_neurons(*s)
        conmats.append(conmat)
    return *conmats, shift_matrix



def id_to_neuron(target_id, target_grp):
    return np.histogram(target_id, bins=range(target_grp.quantity + 1))[0]
