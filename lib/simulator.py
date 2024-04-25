#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""

import numpy as np
import numpy.random as rnd
from dataclasses import dataclass
from cflogger import logger

from params import BaseConfig
from class_lib.population import Population
from lib import pickler as PIC


@dataclass
class Simulator:
    _config: BaseConfig
    _population: Population


    def __post_init__(self):
        """Required for inherent classes."""
        pass


    @property
    def mode(self)->str:
        return self._config.landscape.mode


    def _init_run(self, tag:str, seed:int)->str:
        logger.info(f"Simulate: {tag} with seed: {seed}")
        rnd.seed(seed)
        return tag


    def _save_rate(self, rate:np.ndarray, tag:str) -> None:
        PIC.save_rate(rate, tag, sub_directory=self._config.sub_dir)



    def load_rate(self, tag:str, no_return:bool=False) -> np.ndarray:
        if no_return:
            if PIC.datafile_exists(tag, sub_directory=self._config.sub_dir):
                return True
        # Return a 2D rate data/1D for warmup
        try:
            logger.info(f"Load rate: {tag}")
            rate = PIC.load_rate(tag, sub_directory=self._config.sub_dir)
            return rate
        except FileNotFoundError:
            logger.error("Could not load simulation.")
            return None
