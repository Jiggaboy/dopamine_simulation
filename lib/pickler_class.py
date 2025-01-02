#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger


from  dataclasses import dataclass
import os
import pickle
from pathlib import Path

from params.baseconfig import FIGURE_SUFFIX, FIGURE_ALTERNATIVE_SUFFIX, ANIMATION_SUFFIX
from params.baseconfig import AVG_TAG, SPIKE_TRAIN, FN_RATE
import lib.universal as UNI
from constants import DATA_DIR
#===============================================================================
# CLASS
#===============================================================================

@dataclass
class Pickler:
    config: object


    def save_rate(self, obj: object, postfix: str = None) -> None:
        filename = self.get_filename(postfix)
        filename = self.prepend_dir(filename, self.config.sub_dir)
        logger.info(f"Save rates to {filename}!")
        self.save(filename, obj)


    def load_rate(self, postfix:str=None, skip_warmup:bool=False, exc_only:bool=False) -> object:
        filename = self.get_filename(postfix)
        filename = self.prepend_dir(filename, self.config.sub_dir)

        rate = self.load(filename)
        if skip_warmup:
            # BUG: use self.config.defaultclock_dt here (the monitor one...)
            rate = rate[:, -int(self.config.sim_time):]
        if exc_only:
            rate = rate[:self.config.no_exc_neurons]
        return rate


    def save_avg_rate(self, avgRate, postfix, **kwargs):
        self.save_rate(avgRate, AVG_TAG + postfix)


    def load_average_rate(self, postfix, **kwargs):
        return self.load_rate(AVG_TAG + postfix, **kwargs)


    def _save_spike_train(self, spike_train:object, postfix:str, **kwargs)->None:
        self.save(SPIKE_TRAIN + postfix, spike_train)


    def _load_spike_train(self, postfix, **kwargs) -> object:
        return self.load(SPIKE_TRAIN + postfix, **kwargs)

    def save(self, filename: str, obj: object):
        if obj is None:
            logger.error("No object given. Save cancelled...")
            return
        filename = self.prepend_dir(filename, self.config.sub_dir)
        filename = self.prepend_dir(filename)
        self.create_dir(filename)

        with open(filename, "w+b") as f:
            pickle.dump([obj], f, protocol=-1)


    def load(self, filename: str) -> object:
        filename = self.prepend_dir(filename, self.sub_dir)
        filename = self.prepend_dir(filename)

        with open(filename, "rb") as f:
            return pickle.load(f)[0]

    def get_fig_filename(self, tag:str, format_ = FIGURE_ALTERNATIVE_SUFFIX) -> str:
        fname = self.prepend_dir(tag, directory=FIGURE_DIR)
        return fname + format_


    def save_animation(self, filename:str, animation:object):
        """
        Saves the animation in the subdirectory of the config.
        """
        filename = self.prepend_dir(filename, self.config.sub_dir)
        filename = self.prepend_dir(filename, FIGURE_DIR)
        self.create_dir(filename)
        animation.save(filename + ANIMATION_SUFFIX)


    def save_figure(self, filename:str, figure:object, is_general_figure:bool=False, **kwargs):
        """
        Saves the figure in the subdirectory of the config.
        """
        if not is_general_figure:
            filename = self.prepend_dir(filename, self.config.sub_dir)
        filename = self.prepend_dir(filename, FIGURE_DIR)
        self.create_dir(filename)

        figure.savefig(filename + FIGURE_SUFFIX, **kwargs)
        figure.savefig(filename + FIGURE_ALTERNATIVE_SUFFIX, **kwargs)


    @staticmethod
    def prepend_dir(filename:str, directory:str = DATA_DIR) -> str:
        return os.path.join(directory, filename)


    @staticmethod
    def create_dir(filename:str) -> None:
        """Creates directories such that the filename is valid."""
        path = Path(filename)
        os.makedirs(path.parent.absolute(), exist_ok=True)


    @staticmethod
    def get_filename(postfix: str = None):
        if postfix:
            fname = FN_RATE.replace(".", f"_{postfix}.")
        else:
            fname = FN_RATE
        return fname
