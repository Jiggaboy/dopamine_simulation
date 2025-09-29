#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Histor:
    v0.3: Save the synaptic input as well.


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.3'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

from dataclasses import dataclass
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import brian2
brian2.prefs.core.default_float_dtype = np.float32 # Halves the used disk space.

from brian2 import Network, NeuronGroup, Synapses, StateMonitor
from brian2 import ms

from lib import functimer
# from lib.simulator import Simulator
from params import BaseConfig
from lib.connectivitymatrix import ConnectivityMatrix
from lib import pickler as PIC

#===============================================================================
# CLASS
#===============================================================================

@dataclass
class BrianSimulator:
    _config: BaseConfig
    _population: ConnectivityMatrix

    def __post_init__(self):
        brian2.defaultclock.dt = self._config.defaultclock_dt * ms


    def _init_run(self, tag:str, seed:int, connectivity_matrix:np.ndarray=None)->str:
        logger.info(f"Simulate: {tag} with seed: {seed}")
        rnd.seed(seed)

        brian2.seed(seed)
        brian2.start_scope()
        brian2.seed(seed)
        self.create_network(connectivity_matrix=connectivity_matrix)
        brian2.seed(seed)


    @property
    def mode(self)->str:
        return self._config.landscape.mode

    @functimer
    def run_warmup(self, force:bool=False):
        if not force:
            rate = self.load_rate(self._config.warmup_tag)
            if rate is not None:
                return rate

        self._init_run(self._config.warmup_tag, seed=self._config.warmup_seed)
        rate = self.simulate_warmup()
        self._save_rate(rate[:, -1], self._config.warmup_tag) # Save only the last state to initialize the network again.


    @functimer
    def run_baseline(self, seed:int, force:bool=False):
        tag = self._config.baseline_tag(seed)
        self.run_patch(tag, seed, dop_patch=None, force=force)


    @functimer
    def run_patch(self, tag:str, seed:int, dop_patch:np.ndarray, percent:float = 0., force:bool=False):
        # Run simulation if forced or data file does not exists.
        if not force and self.load_rate(tag, no_return=True):
            return

        # reset and update the connectivity matrix here
        self._population.reset_connectivity_matrix()
        if not dop_patch is None:
            self._population.EE_connections[dop_patch] *= (1. + percent)
        # Creates a network and connects everything
        self._init_run(tag, seed, self._population.connectivity_matrix)
        rate = self.simulate(tag=tag)
        self._save_rate(rate, tag)
        if self._config.save_synaptic_input:
            self._save_synaptic_input(self._monitor.synaptic_input, tag)


    def create_network(self, connectivity_matrix:np.ndarray=None)->None:
        self._neurons = self.create_neuronal_populations()
        # Takes a second
        self._synapses = self.connect(conn_matrix=connectivity_matrix)
        self._monitor = self.monitor()
        self._network = Network(self._neurons, self._synapses, self._monitor)


    def create_neuronal_populations(self)->object:
        N = self._config.no_exc_neurons + self._config.no_inh_neurons
        neurons = NeuronGroup(N, self.neuron_eqs(), method="euler")
        # Init h randomly between 0 and 1
        neurons.h = np.random.uniform(size=N)
        return neurons


    @functimer(logger=logger)
    def connect(self, pre=None, post=None, conn_matrix=None):
        # Setting defaults
        pre = self._neurons if pre is None else pre
        post = self._neurons if post is None else post
        conn_matrix = self._population.connectivity_matrix if conn_matrix is None else conn_matrix

        source, target = conn_matrix.T.nonzero()
        weights = conn_matrix[target, source]
        syn = Synapses(pre, post, model=self.synapse_eqs())
        syn.connect(i=source, j=target)
        syn.w = weights
        return syn


    def monitor(self, neurons=None, dt:float=1*ms) -> StateMonitor:
        neurons = self._neurons if neurons is None else neurons
        return StateMonitor(neurons, ["h", "synaptic_input"], record=True, dt=dt)


    @functimer(logger=logger)
    def simulate_warmup(self):
        self._network.run(self._config.warmup * ms)
        return self._monitor.h


    @functimer(logger=logger)
    def simulate(self, tag:str, force:bool=False, **params):
        self._neurons.h = self.load_rate(self._config.warmup_tag)
        # TODO: What happens if there is no warmup rate?
        self._network.run(self._config.sim_time * ms, **params)
        return self._monitor.h


    def neuron_eqs(self)->str:
        """
        Equations for a rate model with signmoidal transfer function (dF/dt)
        syn_input as sum of exc. and inh. input.
        ext input as mean free noise
        """
        tau = self._config.tau
        tau_noise = self._config.tau_noise
        sigma = self._config.drive.std
        h0 = self._config.transfer_function.offset - self._config.drive.mean
        beta = self._config.transfer_function.slope
        return f"""
            h_max = 1 : 1
            dn/dt = -n / ({tau_noise}*ms) + {sigma}*sqrt(2/({tau_noise}*ms))*xi_n : 1
            dh/dt = -h / ({tau}*ms) + 1 / (1 + exp({beta} * ({h0} - synaptic_input - n))) / ({tau}*ms) :  1
            synaptic_input : 1
        """

    @staticmethod
    def synapse_eqs():
        eqs = """
            w : 1
            synaptic_input_post = w * h_pre : 1 (summed)
          """
        return eqs


    #===============================================================================
    # SAVE AND LOAD RATES
    #===============================================================================
    def _save_rate(self, rate:np.ndarray, tag:str) -> None:
        PIC.save_rate(rate, tag, sub_directory=self._config.sub_dir)


    def load_rate(self, tag:str, no_return:bool=False) -> np.ndarray:
        if no_return:
            return PIC.datafile_exists(tag, sub_directory=self._config.sub_dir)
        # Return a 2D rate data/1D for warmup
        try:
            logger.info(f"Load rate: {tag}")
            rate = PIC.load_rate(tag, sub_directory=self._config.sub_dir)
            return rate
        except FileNotFoundError:
            logger.error("Could not load simulation.")
            return None

    #===============================================================================
    # SAVE SYNAPTIC INPUTS
    #===============================================================================

    def _save_synaptic_input(self, synaptic_input:np.ndarray, tag:str) -> None:
        PIC.save_synaptic_input(synaptic_input, tag, sub_directory=self._config.sub_dir)


def visualise_connectivity(S, figsize:float=(10, 4)):
    """From brian2 tutorials."""
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    plt.plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')
