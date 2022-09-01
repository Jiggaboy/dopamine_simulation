#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:57:15 2022

@author: hauke
"""

import numpy as np
import numpy.random as rnd
from dataclasses import dataclass
import cflogger
log = cflogger.getLogger()

from params import BaseConfig
from custom_class.population import Population
import universal as UNI
from util import pickler as PIC


EXTEND_RATE = True
# EXTEND_RATE = False

WARMUP_SEED = 0


@dataclass
class Simulator:
    _config: BaseConfig
    _population: Population

    def run_baseline(self):
        tags = self._init_run(self._config.baseline_tag)
        rate = self.simulate(self._population, tag=tags, mode=self.mode)
        self._save_rate(rate, tags)


    @property
    def mode(self)->str:
        return self._config.landscape.mode


    @property
    def sub_dir(self)->str:
        return "_".join(self._config.id_)


    def run_warmup(self):
        tags = self._init_run(self._config.warmup_tag, seed=WARMUP_SEED)
        rate = self.simulate(self._population, is_warmup=True)
        self._save_rate(rate, tags)


    def run_baseline(self, seed:int):
        bs_tag = UNI.get_tag_ident(self._config.baseline_tag(seed))
        bs_tag = self._config.baseline_tag(seed)
        tags = self._init_run(bs_tag, seed)
        self._population.reset_connectivity_matrix()
        rate = self.simulate(self._population, tag=tags, mode=self.mode)
        self._save_rate(rate, tags)



    def run_patch(self, dop_patch:np.ndarray, percent:float, tag:str, seed:int):
        tags = self._init_run(tag, seed)

        # reset and update the connectivity matrix here
        self._population.reset_connectivity_matrix()
        self._population.EE_connections[dop_patch] *= (1. + percent)

        rate = self.simulate(self._population, tag=tags, mode=self.mode)
        self._save_rate(rate, tags)



    def _init_run(self, tag:str, seed:int)->str:
        log.info(f"Simulate: {tag} with seed: {seed}")
        rnd.seed(seed)
        return tag

    def _save_rate(self, rate:np.ndarray, tags:str):
        PIC.save_rate(rate, tags, sub_directory=self.sub_dir)


    def _load_rate(self, tags:str):
        return PIC.load_rate(tags, sub_directory=self.sub_dir)





    def init_rate(self, time, tag:str=None, mode:str=None, force:bool=False)->(np.ndarray, int):
        N = self._population.neurons.size
        rate = np.zeros((N, time))
        start = 0

        if force:
            return rate, start

        if EXTEND_RATE:
            try:
                imported_rate = self._load_rate(tag)
                log.info(f"Load rate: {tag}")
            except FileNotFoundError:
                tag = "_".join((mode, "warmup"))
                imported_rate = self._load_rate(tag)
                log.info(f"Load warmup: {tag}")
            start = imported_rate.shape[1] - 1          # account for the connection between preloaded rate and next timestep

            rate[:, :start + 1] = imported_rate

        return rate, start


    def simulate(self, neural_population:Population, **params):
        is_warmup = params.get("is_warmup", False)
        tag = params.get("tag")
        mode = params.get("mode")

        if is_warmup:
            taxis = np.arange(self._config.WARMUP)
            rate, start = self.init_rate(taxis.size, force=True)
        else:
            taxis = np.arange(self._config.sim_time + self._config.WARMUP)

            try:
                rate, start = self.init_rate(taxis.size, tag=tag, mode=mode)
            except ValueError:
                return None

        # Immediate return if nothing to simulate
        if start == taxis.size - 1:
            print("Skipped!")
            return rate

        # Generate GWN as ext. input
        external_input = np.random.normal(self._config.drive.mean, self._config.drive.std, size=rate.T.shape).T

        for t in range(start, taxis.size-1):
            current_rate = rate[:, t]
            r_dot = neural_population.connectivity_matrix @ current_rate + external_input[:, t]  # expensive!!!
            r_dot = UNI.ensure_valid_operation_range(r_dot)

            if t % 500 == 0:
                print(f"{t}: {r_dot.min()} / {r_dot.max()}")

            r_dot = self._config.transfer_function.run(r_dot)
            delta_rate = (- current_rate + r_dot) / self._config.TAU

            rate[:, t + 1] = current_rate + delta_rate
        return rate
