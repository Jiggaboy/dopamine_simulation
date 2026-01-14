#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: 
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

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

from lib.decorator import functimer
from params import config

#===============================================================================
# CONSTANTS
#===============================================================================
N = 100
p = 0.1

tau = config.tau * b2.ms
tau_noise = config.tau_noise * b2.ms
h0  = 20
beta = 0.25
# defaultclock_dt = .5 #ms


#===============================================================================
# MAIN METHOD
#===============================================================================
@functimer
def main():    
    b2.set_device('cpp_standalone', build_on_run=False)
    neurons_pre = b2.NeuronGroup(N, neuron_eqs(std=20, add_stim=True), method="euler")
    neurons_pre.h = np.random.uniform(size=N)*0.25

    neurons_post = b2.NeuronGroup(N, neuron_eqs(std=10), method="euler")
    neurons_post.h = np.random.uniform(size=N)*0.25
    
    synapses_pre = b2.Synapses(neurons_pre, neurons_pre, model=synapse_eqs("rec"))
    synapses_pre.connect(p=0.1)
    synapses_pre.w = 2. ## Adapt here!
    
    synapses = b2.Synapses(neurons_pre, neurons_post, model=synapse_eqs("ff"))
    synapses.connect(p=0.1)
    synapses.w = 20 ## Adapt here!
    
    synapses_post = b2.Synapses(neurons_post, neurons_post, model=synapse_eqs("rec"))
    synapses_post.connect(p=0.1)
    synapses_post.w = 2. ## Adapt here!
    
    monitor_pre  = b2.StateMonitor(neurons_pre, ["h"], record=True, dt=1*b2.ms)
    monitor_post = b2.StateMonitor(neurons_post, ["h", "ff_input"], record=True, dt=1*b2.ms)
    
    
    b2.run(500*b2.ms)
    
    b2.device.build(run=False)  # Compile the code
    results_pre = []
    results_post = []
    # Do 10 runs without recompiling, each time setting group.tau to a new value
    for w in (1, 4, 10):
        for p in (0.8, 1, 1.2):
            b2.device.run(run_args={synapses.w: w*p})
            results_pre.append(monitor_pre.h[:])
            results_post.append(monitor_post.h[:])
    
    
    fig, (ax_pre, ax_post) = plt.subplots(ncols=2, sharey=True)
    
    from matplotlib.colors import TABLEAU_COLORS
    ax = ax_pre
    for m, c in zip(results_pre, TABLEAU_COLORS):
        mean = np.asarray(m).mean(axis=0)
        std = np.asarray(m).std(axis=0)
        ax.plot(mean, c=c)
        # ax.plot(mean+std, c=c, ls="--")
        # ax.plot(mean-std, c=c, ls="--")
        
    ax = ax_post
    for m, c in zip(results_post, TABLEAU_COLORS):
        mean = np.asarray(m).mean(axis=0)
        std = np.asarray(m).std(axis=0)
        ax.plot(mean, c=c)
        # ax.plot(mean+std, c=c, ls="--")
        # ax.plot(mean-std, c=c, ls="--")
    # ax_pre.plot(monitor_pre.h.mean(axis=0))
    # ax_post.plot(monitor_post.h.mean(axis=0))

#===============================================================================
# METHODS
#===============================================================================
def neuron_eqs(std:float, beta:float=0.25, add_stim:bool=False):
    """
    Equations for a rate model with sigmoidal transfer function (dF/dt)
    syn_input as sum of exc. and inh. input.
    ext input as mean free noise
    """
    eqs = f"""
        h_max = 1 : 1
        dn/dt = -n / tau_noise + {std}*sqrt(2/tau_noise)*xi_n : 1
        dh/dt = -h / tau + 1 / (1 + exp(beta * (h0 - (rec_input + ff_input + I_stim) - n))) / tau :  1
        rec_input : 1
        ff_input: 1
        I_stim = 10 * noise_on : 1
    """
    if add_stim:
        stim = """noise_on = int(t > 300*ms and t < 400*ms) : 1"""
    else:
        stim = """noise_on = 0 : 1"""
    return eqs + stim

def synapse_eqs(tag:str):
    if tag not in ("ff", "rec"):
        raise ValueError("No valid synapse given...")
    eqs = f"""
        w : 1 (shared, constant)
        {tag}_input_post = w * h_pre : 1 (summed)
      """
    return eqs

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()