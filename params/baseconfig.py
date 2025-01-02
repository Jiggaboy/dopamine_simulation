#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

from collections import OrderedDict

import lib.universal as UNI
import lib.pickler as PIC
from class_lib import ExternalDrive, Landscape, Synapse, TransferFunction
from params.analysisparams import AnalysisParams


#===============================================================================
#  CONSTANTS
#===============================================================================
# Where would that belong to: Some config - Is there some constants.py? BaseConfig?
# Constants as a class, then BaseConfig inherited?
FN_RATE = "rate.bn"
AVG_TAG = "avg_"
SPIKE_TRAIN = "spike_train_"
SEQ_CROSS_CENTER = "seq_cross_center_"

ANIMATION_SUFFIX = ".gif"
FIGURE_SUFFIX = ".svg"
FIGURE_ALTERNATIVE_SUFFIX = ".png"

#===============================================================================
# CONFIG CLASS
#===============================================================================

class BaseConfig:
    PATH_CONNECTIVITY = "connectivity_matrix"

    # Constants
    TAG_WARMUP = "warmup"
    TAG_BASELINE = "baseline"

    CONSTANT_SEED = True
    warmup_seed = 0

    ##################### Time
    ###### In time steps [a.u.]
    WARMUP = 500
    sim_time = 3000
    TAU = 12. # ms
    tau_noise = 1. # ms
    defaultclock_dt = .5 #ms

    ##################### Patches
    center_range = OrderedDict({})

    radius = (6, 12, 18)
    AMOUNT_NEURONS = (10, 50, 100)
    PERCENTAGES = (.3, .2, .1)


    #####################
    rows = 80
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
        logger.debug("Retrieve Config ID")
        main = self.landscape.mode, *[str(s) for s in self.landscape.params.values()],  str(self.rows)
        connection = str(self.landscape.connection_probability), str(self.synapse.weight), str(self.synapse.EI_factor)
        gaussian = "std", str(self.landscape.stdE), str(self.landscape.stdI), str(self.landscape.shift)
        drive = "drive", str(self.drive.mean), str(self.drive.std)
        transfer = "transfer", str(self.transfer_function.offset), str(self.transfer_function.slope)

        try:
            return *main, *gaussian, *connection, *drive, *transfer
        except Exception:
            return *main, *connection, *drive, *transfer


    @property
    def sub_dir(self)->str:
        return "_".join(self.id_)


    @property
    def no_exc_neurons(self)->int:
        return self.rows**2


    @property
    def no_inh_neurons(self)->int:
        return self.rows**2 // 4


    def __init__(self):
        self.__post_init__()


    def __post_init__(self):

        if hasattr(self, "base"):
            logger.info("Set new base for landscape.")
            self.landscape.params["base"] = self.base

        self.analysis = AnalysisParams()
        self.coordinates = UNI.get_coordinates(self.rows)
        logger.info(f"Config name: {self.__class__.__name__}")
        if self.landscape:
            logger.info(f"Landscape: {self.landscape.params.values()} on {self.rows} rows.")
            logger.info(f"Landscape info: {self.landscape.info.values()} with Synapses: {self.synapse.values}.")

        if hasattr(self.analysis, "dbscan_controls"):
            self.analysis.dbscan_controls.detection_spots = self._add_detection_spots()

        self.AMOUNT_NEURONS = UNI.make_iterable(self.AMOUNT_NEURONS)
        self.PERCENTAGES = UNI.make_iterable(self.PERCENTAGES)
        self.radius = UNI.make_iterable(self.radius)

    def _add_detection_spots(self) -> None:
        return []


    def __str__(self) -> str:
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


    def get_baseline_tag_from_tag(self, tag:str):
        _, seed = UNI.split_seed_from_tag(tag)
        return self.baseline_tag(seed)


    def path_to_connectivity_matrix(self):
        return UNI.get_tag_ident(self.PATH_CONNECTIVITY,
                                 self.landscape.mode,
                                 self.rows,
                                 *self.landscape.params.values(),
                                 *self.landscape.info.values(),
                                 *self.synapse.values) + ".bn"


    def get_all_tags(self, patchnames:tuple=None, radius:tuple=None, amount:tuple=None, synaptic_fraction=None, weight_change:tuple=None, seeds:tuple=None):
        patchnames = patchnames or self.center_range
        patchnames = UNI.make_iterable(patchnames)
        radius = radius or self.radius
        amount = amount or self.AMOUNT_NEURONS
        weight_change = self.PERCENTAGES if weight_change is None else UNI.make_iterable(weight_change)

        tags = []
        seeds, method = self._seeds_and_method(seeds, tags)

        for name in patchnames:
        # patch_range = {k: v for k, v in self.center_range.items() if k in patchnames}
        # for name, center in patch_range.items():
            for r in UNI.make_iterable(radius):
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
