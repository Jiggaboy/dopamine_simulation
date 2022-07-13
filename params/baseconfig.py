#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:38:45 2022

@author: hauke
"""


import logging
log = logging.getLogger()

import numpy as np
from collections import namedtuple, OrderedDict

import universal as UNI
import util.pickler as PIC
from custom_class import ExternalDrive, Landscape, Plasticity, Synapse, TransferFunction


class BaseConfig:


    POPULATION_FILENAME = "Population_{}_{}.bn"
    PATH_CONNECTIVITY = "connectivity_matrix_{}_{}.bn"

    # Constants
    TAG_WARMUP = "warmup"
    TAG_BASELINE = "baseline"

    CONSTANT_SEED = True

    ##################### Time
    ###### In time steps [a.u.]
    WARMUP = 500
    sim_time = 15000.
    TAU = 12.


    ##################### Patches
    center_range = OrderedDict({
        "repeater": (17, 34),
        "starter": (43, 68),
        # "starter2": (42, 67),
        # "starter3": (44, 67),
        # "starter4": (42, 69),
        # "starter5": (44, 69),
        "linker": (16, 56),
        #"in-activator": (66, 34),
        "edge-activator": (63, 34),
        #"out-activator": (59, 34),
        "in": (35, 18),
        "edge": (35, 22),
        "out": (35, 26),
    })


    RADIUSES = (6, 12, 18)
    AMOUNT_NEURONS = (10, 50, 100)
    PERCENTAGES = (.3, .2, .1)
    P_synapses = (1., .8, .6)


    #####################
    rows = 70
    plasticity = Plasticity(rate=.1, cap=2.)
    synapse = Synapse(weight=2., EI_factor=6.5)
    transfer_function = TransferFunction(50., .5)
    drive = ExternalDrive(20., 20.)

    landscape = None


    @property
    def warmup_tag(self)->str:
        return UNI.get_tag_ident(self.landscape.mode, self.TAG_WARMUP)


    @property
    def baseline_tag(self)->str:
        return UNI.get_tag_ident(self.landscape.mode, self.TAG_BASELINE)


    @property
    def info(self)->OrderedDict:
        params = OrderedDict()
        params["rows"] = self.rows
        params.update(self.landscape.info)
        return params


    @property
    def id_(self)->tuple:
        return self.landscape.mode, str(self.rows)


    @property
    def sub_dir(self)->str:
        return "_".join(self.id_)


    def __init__(self):
        self.__post_init__()


    def __post_init__(self):
        log.info("\n".join(("Configuration:", f"Rows: {self.rows}", f"Landscape: {self.landscape}")))



    def __str__(self):
        props = {
            "landscape": self.landscape,
            "transfer function": self.transfer_function,
            "synapse": self.synapse,
            "drive": self.drive,
            "Time": (self.WARMUP, self.sim_time)
        }
        return str(props)


    def path_to_connectivity_matrix(self):
        return self.PATH_CONNECTIVITY.format(self.landscape.mode, self.rows)


    def get_all_tags(self, patchnames:tuple=None, radius:tuple=None, amount:tuple=None, synaptic_fraction=None, weight_change:tuple=None):
        patchnames = patchnames or self.center_range
        radius = radius or self.RADIUSES
        amount = amount or self.AMOUNT_NEURONS
        synaptic_fraction = synaptic_fraction or self.P_synapses
        weight_change = weight_change or self.PERCENTAGES

        tags = []
        for name in patchnames:
            for r in radius:
                for a in amount:
                    for s in synaptic_fraction:
                        for w in weight_change:
                            tags.append(UNI.get_tag_ident(name, r, a, s, int(w*100)))
        return tags
    
    
    def get_center(self, tag:str)->tuple:
        return self.center_range[tag]
    
    
    def find_tags(self, tags:tuple)->list:
        """
        Finds all the tags in the config starting with element in tags.
        """
        found_tags = []
        for tag in tags:
            found_tags.extend([t for t in self.get_all_tags() if t.startswith(tag)])
        return found_tags
    

    def save(self, subdir:str=None):
        PIC.save("config.txt", str(self), sub_directory=subdir)
