#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:57:15 2022

@author: Hauke Wernecke
"""

import numpy as np
import numpy.random as rnd
from dataclasses import dataclass
import cflogger
log = cflogger.getLogger()

from params import BaseConfig
from custom_class.population import Population
import lib.universal as UNI
from lib import pickler as PIC
from lib import functimer


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


    @property
    def sub_dir(self)->str:
        return "_".join(self._config.id_)


    def run_warmup(self, **sim_kwargs):
        tags = self._init_run(self._config.warmup_tag, seed=self._config.warmup_seed)
        rate = self.simulate(self._population, is_warmup=True, tag=self._config.warmup_tag, mode=self.mode, **sim_kwargs)
        self._save_rate(rate, tags)


    def run_baseline(self, seed:int, **sim_kwargs):
        bs_tag = self._config.baseline_tag(seed)
        tags = self._init_run(bs_tag, seed)
        self._population.reset_connectivity_matrix()
        rate = self.simulate(self._population, tag=tags, mode=self.mode, **sim_kwargs)

        # import matplotlib.pyplot as plt
        # plt.figure("rate_ode")
        # for r in rate[:16]:
        #     plt.plot(r)
        self._save_rate(rate, tags)



    def run_patch(self, dop_patch:np.ndarray, percent:float, tag:str, seed:int, **sim_kwargs):
        tags = self._init_run(tag, seed)

        # reset and update the connectivity matrix here
        self._population.reset_connectivity_matrix()
        self._population.EE_connections[dop_patch] *= (1. + percent)

        rate = self.simulate(self._population, tag=tags, mode=self.mode, **sim_kwargs)
        self._save_rate(rate, tags)



    def _init_run(self, tag:str, seed:int)->str:
        log.info(f"Simulate: {tag} with seed: {seed}")
        rnd.seed(seed)
        return tag

    def _save_rate(self, rate:np.ndarray, tags:str):
        PIC.save_rate(rate, tags, sub_directory=self.sub_dir)


    def _load_rate(self, tags:str):
        return PIC.load_rate(tags, sub_directory=self.sub_dir)



    def load_warmup_rate(self, force:bool):
        # Default 1D array of inital values
        def default_initial_values():
            N = self._population.neurons.size
            return np.zeros(N)

        if force:
            return default_initial_values()

        try:
            # Return a 2D rate data
            warmup_rate = self._load_rate(self._config.warmup_tag)
            log.info(f"Load warmup: {self._config.warmup_tag}")
            return warmup_rate
        except FileNotFoundError:
            return default_initial_values()


    def load_initial_values_from_warmup_rate(self, tag:str, force:bool):
        # Return a 1D vector of initial values
        def default_initial_values():
            warmup_rate = self._load_rate(self._config.warmup_tag)
            return warmup_rate[:, -1]

        if force:
            return default_initial_values()

        try:
            # Return a 2D rate data
            rate = self._load_rate(tag)
            log.info(f"Load rate: {tag}")
            return rate
        except FileNotFoundError:
            return default_initial_values()


    @staticmethod
    def ODE(t, y, W:np.ndarray, tau:float, noise:np.ndarray, transfer_function):
        input_ = W.dot(y) + noise[int(t)]
        drdt = transfer_function.run(input_)
        return (1 / tau) * (-y + drdt)


    @functimer(logger=log)
    def simulate(self, neural_population:Population, **params):
        is_warmup = params.get("is_warmup", False)
        tag = params.get("tag")
        force = params.get("force")

        if is_warmup:
            taxis = np.arange(self._config.WARMUP)
            rate = self.load_warmup_rate(force)
        else:
            taxis = np.arange(self._config.sim_time)
            rate = self.load_initial_values_from_warmup_rate(tag, force)

        if rate.ndim == 2:
            return rate

        # Generate GWN as ext. input -> Using that size and the transpose ensures the same noise for the first N timesteps
        external_input = np.random.normal(self._config.drive.mean, self._config.drive.std, size=(taxis.size, rate.size)).T

        args=(neural_population.connectivity_matrix, self._config.TAU, external_input.T, self._config.transfer_function)


        euler_rate = np.zeros((self._population.neurons.size, taxis.size))
        euler_rate[:, 0] = rate
        for t in taxis[:-1]:
            euler_rate[:, t+1] = euler_rate[:, t] + self.ODE(t, euler_rate[:, t], *args)        #

        return euler_rate
