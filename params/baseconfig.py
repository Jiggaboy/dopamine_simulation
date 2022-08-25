#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:38:45 2022

@author: hauke
"""


import cflogger
logger = cflogger.getLogger()

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
    center_range = OrderedDict({})


    RADIUSES = (6, 12, 18)
    AMOUNT_NEURONS = (10, 50, 100)
    PERCENTAGES = (.3, .2, .1)


    #####################
    rows = 70
    plasticity = Plasticity(rate=.1, cap=2.)
    synapse = Synapse(weight=2., EI_factor=6.5)
    transfer_function = TransferFunction(50., .5)
    drive = ExternalDrive(20., 20., seeds=(0, 1))

    landscape = None



    @property
    def warmup_tag(self)->str:
        return UNI.get_tag_ident(self.landscape.mode, self.TAG_WARMUP)


    @property
    def baseline_tags(self)->str:
        return [UNI.get_tag_ident(self.landscape.mode, self.TAG_BASELINE, seed) for seed in self.drive.seeds]
    
    
    @property
    def simulation_seeds(self)->tuple:
        return self.drive.seeds


    @property
    def info(self)->OrderedDict:
        params = OrderedDict()
        params["rows"] = self.rows
        params.update(self.landscape.info)
        return params


    @property
    def id_(self)->tuple:
        logger.info("Retrieve Config ID")
        main = self.landscape.mode, str(self.rows)
        connection = str(self.landscape.connection_probability), str(self.synapse.weight)
        gaussian = "std" + str(self.landscape.stdE) + str(self.landscape.stdI)
        try:
            return *main, gaussian, *connection
        except Exception:
            return *main, *connection
        

    @property
    def sub_dir(self)->str:
        return "_".join(self.id_)


    @property
    def no_exc_neurons(self)->int:
        return self.rows**2
    
    def __init__(self):
        self.__post_init__()


    def __post_init__(self):
        logger.info("\n".join(("Configuration:", f"Rows: {self.rows}", f"Landscape: {self.landscape}")))



    def __str__(self):
        props = {
            "landscape": self.landscape,
            "transfer function": self.transfer_function,
            "synapse": self.synapse,
            "drive": self.drive,
            "Time": (self.WARMUP, self.sim_time)
        }
        return str(props)

    
    def baseline_tag(self, seed:int)->str:
        return UNI.get_tag_ident(self.landscape.mode, self.TAG_BASELINE, seed)
                

    def path_to_connectivity_matrix(self):
        return self.PATH_CONNECTIVITY.format(self.landscape.mode, self.rows)


    def get_all_tags(self, patchnames:tuple=None, radius:tuple=None, amount:tuple=None, synaptic_fraction=None, weight_change:tuple=None, seeds:tuple=None):
        patchnames = patchnames or self.center_range
        patchnames = UNI.make_iterable(patchnames)
        radius = radius or self.RADIUSES
        amount = amount or self.AMOUNT_NEURONS
        weight_change = weight_change or self.PERCENTAGES
        weight_change = self.PERCENTAGES if weight_change is None else weight_change
        
        tags = []
        seeds, method = self._seeds_and_method(seeds, tags)

        for name in patchnames:
            for r in radius:
                for a in amount:
                    for w in weight_change:
                        tmp = [UNI.get_tag_ident(name, r, a, int(w*100), s) for s in seeds]
                        method(tmp)
        return tags
    
    
    def _seeds_and_method(self, seeds:(int, tuple, str), l:list):
        """
        Either takes a subset of seeds or all seeds.
        Determines the method for the list l.
        """
        method = l.extend
        seed_iter = self.drive.seeds if seeds is None else seeds
        logger.info(f"seed_iter: {seed_iter}")
        if seeds == "all":
            method = l.append
            seed_iter = self.drive.seeds
        return UNI.make_iterable(seed_iter), method
            
    
    def get_center(self, tag:str)->tuple:
        return self.center_range[tag]
    
    
    def find_tags(self, tags:tuple)->list:
        """
        Finds all the tags in the config starting with element in tags.
        """
        tags = UNI.make_iterable(tags)
        found_tags = []
        for tag in tags:
            found_tags.extend([t for t in self.get_all_tags() if t.startswith(tag)])
        return found_tags
    

    def save(self, subdir:str=None):
        PIC.save("config.txt", str(self), sub_directory=subdir)
