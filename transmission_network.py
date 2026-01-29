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
from matplotlib import rcParams
import numpy as np

from lib.decorator import functimer
from params import config
from plot.constants import COLOR_MAP_ACTIVITY, NORM_ACTIVITY, COLOR_MAP_DIFFERENCE, cm
from matplotlib.colors import TABLEAU_COLORS
#===============================================================================
# PLOTTING
#===============================================================================
rcParams["font.size"] = 8
rcParams["figure.figsize"] = (17.6*cm, 6*cm)
rcParams["legend.fontsize"] = 7
rcParams["legend.markerscale"] = 0.6
rcParams["legend.handlelength"] = 1.25
rcParams["legend.columnspacing"] = 1
rcParams["legend.handletextpad"] = 1
rcParams["legend.labelspacing"] = .1
rcParams["legend.borderpad"] = .25
rcParams["legend.handletextpad"] = .5
rcParams["legend.framealpha"] = 1
rcParams["axes.labelpad"] = 2
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False

legend_kwargs = {"ncol": 2, "loc": "upper center"}
#===============================================================================
# CONSTANTS
#===============================================================================
simtime = 500
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
    
    
    b2.run(100*b2.ms)
    b2.run(simtime*b2.ms)
    
    b2.device.build(run=False)  # Compile the code
    
    weights = (1, 3., 9)
    percentages = (1.2, 1, .8)
    results_pre = np.zeros((len(weights), len(percentages)), dtype=object)
    results_post = np.zeros((len(weights), len(percentages)), dtype=object)
    
    # Do 10 runs without recompiling, each time setting group.tau to a new value
    for w, weight in enumerate(weights):
        for p, percent in enumerate(percentages):
            b2.device.run(run_args={synapses.w: weight*percent})
            results_pre[w, p] = monitor_pre.h[:, 100:]
            results_post[w, p] = monitor_post.h[:, 100:]
    
    
    
    ## Proper plotting
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=3)
    fig.subplots_adjust(
        left=0.0,
        right=0.96,
        bottom=0.2,
        top=0.90,
        wspace=0.1,
    )
    
    ax_pre = fig.add_subplot(gs[1])
    ax_pre.set_title("Source Network")
    ax_pre.set_xlabel("Time [ms]")
    ax_pre.set_ylabel("Rate")
    ax_pre.set_xticks(np.arange(0, 600, 100))
    ax_pre.set_xlim((0, simtime))
    ax_pre.set_yticks((0, 0.5, 1))
    ax_pre.set_ylim((0, 1))
    
    ax_post = fig.add_subplot(gs[2])
    ax_post.set_title("Target Network")
    ax_post.set_xlabel("Time [ms]")
    ax_post.set_xticks(np.arange(0, 600, 100))
    ax_post.set_xlim((0, simtime))
    ax_post.set_yticks((0, 0.5, 1))
    ax_post.set_ylim((0, 1))
    ax_post.tick_params(labelleft=False)
    
    time = np.arange(simtime)
    ax = ax_pre
    for w, (weight, color) in enumerate(zip(weights, TABLEAU_COLORS)):
        mean = results_pre[w, 0].mean(axis=0)
        ax.plot(time, mean, c=color, ls="--", alpha=1, label=f"{1-percentages[w]:+.0%}")

        mean = results_pre[w, 1].mean(axis=0)
        ax.plot(time, mean, c=color, ls="-", alpha=.8, label=f"{1-percentages[w]:+.0%}")
        
        mean = results_pre[w, 2].mean(axis=0)
        ax.plot(time, mean, c=color, ls="--", alpha=.6, label=f"{1-percentages[w]:+.0%}")
    ax.legend(ncols=3, handletextpad=0.1, labelspacing=.1,
               markerscale=0.75, loc="upper center", 
        handlelength = 1.,
        borderpad = 0.2, columnspacing=1)
        
    ax = ax_post
    for w, (weight, color) in enumerate(zip(weights, TABLEAU_COLORS)):
        mean = results_post[w, 0].mean(axis=0)
        ax.plot(time, mean, c=color, ls="--", alpha=1)
        
        mean = results_post[w, 1].mean(axis=0)
        ax.plot(time, mean, c=color, ls="-", alpha=.8)
        
        mean = results_post[w, 2].mean(axis=0)
        ax.plot(time, mean, c=color, ls="--", alpha=.6)
        
    import lib.pickler as PIC
    PIC.save_figure("transmission_raw", fig, transparent=True)
    return

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
        I_stim = 8 * noise_on : 1
    """
    if add_stim:
        stim = """noise_on = int(t > 200*ms and t < 250*ms) - int(t > 450*ms and t < 500*ms)  : 1"""
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