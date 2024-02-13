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

import numpy as np
import matplotlib.pyplot as plt

import brian2
from brian2 import Network, NeuronGroup, Synapses, StateMonitor
from brian2 import ms

from lib import functimer
from lib.simulator import Simulator

#===============================================================================
# CLASS
#===============================================================================

class BrianSimulator(Simulator):

    def __post_init__(self):
        super().__post_init__()
        brian2.defaultclock.dt = self._config.defaultclock_dt * ms


    def _init_run(self, tag:str, seed:int, connectivity_matrix:np.ndarray=None)->str:
        super()._init_run(tag, seed)
        brian2.start_scope()
        self.create_network(connectivity_matrix=connectivity_matrix)



    @functimer
    def run_warmup(self, force:bool=False, **sim_kwargs):
        if not force:
            rate = self.load_rate(self._config.warmup_tag)
            if rate is not None:
                return rate

        self._init_run(self._config.warmup_tag, seed=self._config.warmup_seed)
        print("run_warmup (if this never shows something, remove sim_kwargs): ", **sim_kwargs)
        rate = self.simulate_warmup()
        self._save_rate(rate[:, -1], self._config.warmup_tag)


    @functimer
    def run_baseline(self, seed:int, force:bool=False, **sim_kwargs):
        tag = self._config.baseline_tag(seed)
        self.run_patch(tag, seed, dop_patch=None, force=force)


    @functimer
    def run_patch(self, tag:str, seed:int, dop_patch:np.ndarray, percent:float = 0., force:bool=False, **sim_kwargs):
        if not force:
            rate = self.load_rate(tag, no_return=True)
            if rate is not None:
                return rate

        # reset and update the connectivity matrix here
        self._population.reset_connectivity_matrix()
        if not dop_patch is None:
            self._population.EE_connections[dop_patch] *= (1. + percent)
        # Creates a network and connects everything
        self._init_run(tag, seed, self._population.connectivity_matrix)
        print("run_patch (if this never shows something, remove sim_kwargs): ", **sim_kwargs)
        rate = self.simulate(tag=tag, **sim_kwargs)
        self._save_rate(rate, tag)


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
        # Determine x and y positions of each neuron (assumption: square grid)
        neurons.x = "i % sqrt(N)"
        neurons.y = "i // sqrt(N)"
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
        self._network.run(self._config.WARMUP * ms)
        return self._monitor.h


    @functimer(logger=logger)
    def simulate(self, tag:str, force:bool=False, **params):
        self._neurons.h = self.load_rate(self._config.warmup_tag)
        self._network.run(self._config.sim_time * ms)
        return self._monitor.h



    def neuron_eqs(self)->str:
        """
        Equations for a rate model with signmoidal transfer function (dF/dt)
        syn_input as sum of exc. and inh. input.
        ext input as mean free noise
        """
        tau = self._config.TAU
        tau_noise = self._config.tau_noise
        sigma = self._config.drive.std
        h0 = self._config.transfer_function.offset - self._config.drive.mean
        beta = self._config.transfer_function.slope
        return f"""
            h_max = 1 : 1
            tau_n = 1.*ms : second
            dn/dt = -n / ({tau_noise}*ms) + {sigma}*sqrt(2/({tau_noise}*ms))*xi_n : 1
            dh/dt = -h / ({tau}*ms) + 1 / (1 + exp({beta} * ({h0} - synaptic_input - n))) / ({tau}*ms) :  1
            synaptic_input : 1
            x : 1
            y : 1
        """

    @staticmethod
    def synapse_eqs():
        eqs = """
            w : 1
            synaptic_input_post = w * h_pre : 1 (summed)
          """
        return eqs


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




#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    main()
