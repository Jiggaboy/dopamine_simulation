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
__version__ = '0.2'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
import numpy as np
import matplotlib.pyplot as plt

import brian2
brian2.prefs.core.default_float_dtype = np.float32

from brian2 import Network, NeuronGroup, Synapses, StateMonitor
from brian2 import ms


#===============================================================================
# CLASS
#===============================================================================

class BrianSimulator:

    def __init__(self, dt:float=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt


    @property
    def dt(self):
        return brian2.defaultclock.dt


    @dt.setter
    def dt(self, _dt:float):
        brian2.defaultclock.dt = _dt * ms


    def run(self, time:float):
        self._network.run(time * ms)
        return self._monitor.h


    def create_network(self, NE:int, NI:int, connectivity_matrix:np.ndarray, **neuron_kwargs)->None:
        self._neurons = self.create_neuronal_populations(NE, NI, **neuron_kwargs)
        # Takes a second
        self._synapses = self.connect(conn_matrix=connectivity_matrix)
        self._monitor = self.monitor()
        self._network = Network(self._neurons, self._synapses, self._monitor)


    def create_neuronal_populations(self, NE:int, NI:int, **neuron_kwargs)->object:
        N = NE + NI
        neurons = NeuronGroup(N, self.neuron_eqs(**neuron_kwargs), method="euler")
        # Init h randomly between 0 and 1
        neurons.h = np.random.uniform(size=N)
        return neurons


    def connect(self, conn_matrix:np.ndarray) -> Synapses:
        source, target = conn_matrix.T.nonzero()
        weights = conn_matrix[target, source]
        syn = Synapses(self._neurons, self._neurons, model=self.synapse_eqs())
        syn.connect(i=source, j=target)
        syn.w = weights
        return syn


    def monitor(self, neurons=None, dt:float=1*ms) -> StateMonitor:
        neurons = self._neurons if neurons is None else neurons
        return StateMonitor(neurons, ["h", "synaptic_input"], record=True, dt=dt)


    def reset_rates(self):
        self._network.remove(self._monitor)
        self._monitor = self.monitor()
        self._network.add(self._monitor)


    def neuron_eqs(self, **neuron_kwargs)->str:
        """
        Equations for a rate model with signmoidal transfer function (dF/dt)
        syn_input as sum of exc. and inh. input.
        ext input as mean free noise
        """
        tau_membrane = neuron_kwargs.get("tau_membrane", 12)
        tau_noise = neuron_kwargs.get("tau_noise", .1)
        sigma = neuron_kwargs.get("sigma", 30)
        h0 = neuron_kwargs.get("h0", 40)
        slope = neuron_kwargs.get("slope", .25)
        return f"""
            h_max = 1 : 1
            tau_n = 1.*ms : second
            dn/dt = -n / ({tau_noise}*ms) + {sigma}*sqrt(2/({tau_noise}*ms))*xi_n : 1
            dh/dt = -h / ({tau_membrane}*ms) + 1 / (1 + exp({slope} * ({h0} - synaptic_input - n))) / ({tau_membrane}*ms) :  1
            synaptic_input : 1
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
