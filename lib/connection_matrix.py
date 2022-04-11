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
from lib.move import move

from dataclasses import dataclass

@dataclass
class Group:
    rows: int
    std: float

    def __post_init__(self):
        self.quantity = self.rows ** 2

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



def set_seed(seed):
    if seed is None:
        np.random.seed()
        seed = np.random.randint(1000)
    else:
        np.random.seed(seed)


def get_shift_matrix(landscape:str, excitatory_group:Group)->np.ndarray:
    return cl.__dict__[landscape.mode](excitatory_group.rows, landscape.params)
####### Dpereciated: direction 0 corresponds to a shift righwards.
    # if landscape.is_asymmetric:
    #     shift = cl.__dict__[landscape.mode](excitatory_group.rows, landscape.params)
    # else:
    #     shift = np.zeros(excitatory_group.quantity)
    # return shift


def EI_networks(landscape, nrowE, **kwargs):
# def EI_networks(landscape, nrowE, p, stdE, stdI, shift=0, seed=None, **kwargs):
    grp_E = Group(nrowE, landscape.stdE)
    grp_I = Group(nrowE // 2, landscape.stdI)

    set_seed(landscape.seed)

    find_targets = lcrn.independent_targets if landscape.is_independent else lcrn.lcrn_gauss_targets

    shift_matrix = get_shift_matrix(landscape, grp_E)



    conmatEE = np.zeros((grp_E.quantity, grp_E.quantity))
    conmatEI = np.zeros((grp_E.quantity, grp_I.quantity))
    conmatIE = np.zeros((grp_I.quantity, grp_E.quantity))
    conmatII = np.zeros((grp_I.quantity, grp_I.quantity))

    EE_setup = conmatEE, grp_E, grp_E, landscape.is_asymmetric, shift_matrix
    EI_setup = conmatEI, grp_E, grp_I, True, None
    IE_setup = conmatIE, grp_I, grp_E, True, None
    II_setup = conmatII, grp_I, grp_I, False, None
    setups = (EE_setup, EI_setup, IE_setup, II_setup)


    def find_neurons(source, target, allow_self_connections, shift):
        for idx in range(source.quantity):
            cnn = Connection(idx, source, target, landscape.connection_probability, allow_self_connections=allow_self_connections)


            direction = None
            if shift is not None:
                # source_id = move(idx, shift_matrix[idx], nrowE)
                # cnn.idx = source_id
                # allow_self_connections = False
                direction = shift_matrix[idx]

            postsynaptic_ids = find_targets(*cnn.get_all(), direction=direction, shift=landscape.shift)
            # if shift is not None:
            #     tmp_postsynaptic_ids = tmp_postsynaptic_ids[tmp_postsynaptic_ids != idx]
            # postsynaptic_ids = tmp_postsynaptic_ids[:cnn.no_of_targets]
            # postsynaptic_ids = find_targets(*cnn.get_all(), direction=direction, shift=landscape.shift)




        # #####################
        #     postsynaptic_ids = find_targets(*cnn.get_all(), direction=direction)

        #     if shift is not None:
        #         postsynaptic_ids = move(postsynaptic_ids, shift_matrix[idx], nrowE)
                # postsynaptic_ids = postsynaptic_ids[postsynaptic_ids != idx]
        #         # fill_diagonal? https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html
        # ########################

            conmat[idx] = id_to_neuron(postsynaptic_ids, target)
        return conmat


    for conmat, *s in setups:
        conmat = find_neurons(*s)

    return conmatEE, conmatEI, conmatIE, conmatII, shift_matrix


    # source_group = grp_E
    # for idx in range(source_group.quantity):
    #     cnn = Connection(idx, source_group, None, p)

    #     target_grps = grp_E, grp_I
    #     self_conns = is_asymetric, True
    #     conmats = conmatEE, conmatEI
    #     shifts = ll, None
    #     for target_grp, self_conn, conmat, shift in zip(target_grps, self_conns, conmats, shifts):
    #         cnn.target = target_grp
    #         cnn.allow_self_connections = self_conn

    #         targets = find_targets(*cnn.get_all())

    #         if shift is not None:
    #             targets = move(targets, ll[idx], nrowE)
    #             targets = targets[targets != idx]

    #         conmat[idx] = id_to_neuron(targets, target_grp)




    # source_group = grp_I
    # for idx in range(source_group.quantity):
    #     cnn = Connection(idx, grp_I, None, p)

    #     target_grps = grp_E, grp_I
    #     self_conns = True, False
    #     conmats = conmatIE, conmatII
    #     for target_grp, self_conn, conmat in zip(target_grps, self_conns, conmats):
    #         cnn.target = target_grp
    #         cnn.allow_self_connections = self_conn

    #         targets = find_targets(*cnn.get_all())

    #         conmat[idx] = id_to_neuron(targets, target_grp)

    # return np.array(conmatEE), np.array(conmatEI), np.array(conmatIE), np.array(conmatII), ll


def id_to_neuron(target_id, target_grp):
    return np.histogram(target_id, bins=range(target_grp.quantity + 1))[0]
