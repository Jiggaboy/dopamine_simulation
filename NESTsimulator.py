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

import nest
from simulator import Simulator
from params import BaseConfig
from custom_class.population import Population
import universal as UNI
from lib import pickler as PIC


EXTEND_RATE = True
# EXTEND_RATE = False

WARMUP_SEED = 0


class NESTSimulator(Simulator):

    _neuron_model = "sigmoid_rate_ipn"
    _generator_model = "step_rate_generator"

    """
    rate: Rate (unitless)
    tau: ms; Time constant of rate dynamics
    mu: Mean input
    sigma: Noise parameter
    g: Gain parameter
    beta: Slope parameter
    theta: Threshold
    """

    def __post_init__(self):
        print("Run post init")


    def _create_network(self, ):
        neurons = self._create_neural_population()
        recorder = self._record_neurons(neurons)
        return neurons, recorder
        """
        dt = 1.
        time_line = np.arange(dt, self._config.WARMUP +  + dt, dt)
        generator = nest.Create(self._generator_model, n=n)
        for i in range(n):
            noise = np.random.normal(loc=ext_drive_mu, scale=ext_drive_sigma, size=time_line.size)
            generator[i].set(
                amplitude_times = time_line,
                amplitude_values = noise
            )

        nest.Connect(generator, neurons, conn_spec={'rule': 'one_to_one'}, syn_spec={'synapse_model': "rate_connection_delayed", "delay": dt})

        return neurons, generator, mm_neurons, mm_generator
        """

    def _create_rate_recorder(self, interval:float=1.):
        return nest.Create('multimeter', params={'record_from': ['rate'], 'interval': interval})


    def _create_neural_population(self):
        neurons = nest.Create(self._neuron_model, n=self._config.no_exc_neurons + self._config.no_inh_neurons)
        return neurons


    def _record_neurons(self, neurons, **recorder_params):
        recorder = self._create_rate_recorder(**recorder_params)
        nest.Connect(recorder, neurons)
        return recorder


    def _connect_network(self, neurons, connectivity_matrix)->None:
        log.info("Connect Neurons.")
        # self._population.connectivity_matrix is the setup for warmup and baseline
        for i, pre in enumerate(neurons):
            targets, weights = self._get_targets_and_weigths(neurons, connectivity_matrix, i)
            pre_vector = np.full(fill_value=pre, shape=len(targets))
            self._connect_rate_neurons(pre_vector, targets, weights)



    @staticmethod
    def _connect_rate_neurons(pre:np.ndarray, post:np.ndarray, weights:np.ndarray):
        nest.Connect(pre, post, conn_spec='one_to_one', syn_spec={'synapse_model': "rate_connection_instantaneous", "weight": weights})

    @staticmethod
    def _get_targets_and_weigths(neurons:np.ndarray, connectivity_matrix:np.ndarray, target_col:int):
        weights = connectivity_matrix[:, target_col]

        # To only connect pairs with a nonzero weight, we use array indexing to extract the weights and targets (post-synaptic) neurons.
        nonzero_indices = np.where(weights != 0)[0]
        weights = weights[nonzero_indices]
        targets = neurons[nonzero_indices]

        return targets, weights



    def run_baseline(self):
        tags = self._init_run(self._config.baseline_tag)
        rate = self.simulate(self._population, tag=tags, mode=self.mode)
        self._save_rate(rate, tags)


    def run_warmup(self):
        nest.ResetKernel()
        neurons, recorder = self._create_network()
        self._connect_network(neurons, self._population.connectivity_matrix)
        nest.Simulate(self._config.WARMUP)
        print(recorder.events)

        return
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
