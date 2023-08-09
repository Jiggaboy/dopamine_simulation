# -*- coding: utf-8 -*-
"""
This script runs several simulations defined by the configuration.
"""
import cflogger
log = cflogger.getLogger()

import matplotlib.pyplot as plt
import numpy as np

from lib import functimer
from brian2 import NeuronGroup, Synapses, StateMonitor, run, defaultclock, ms, second, Function, meter, start_scope, plot, figure, rand


@functimer(logger=log)
def brian():
    defaultclock.dt = .01*ms
    print(defaultclock.dt)

    start_scope()
    neurons = NeuronGroup(100, eqs(sigma=20.), method = "euler")

    statemon_exc = StateMonitor(neurons, ["n"], record=True, dt=.1*ms)


    run(1000*ms)

    print("STD: ", statemon_exc.n[0].std())
    print("STD: ", statemon_exc.n.std())

    plt.figure("noise")
    for n in statemon_exc.n:
        plt.plot(n)

    plt.figure("corr")
    for n in statemon_exc.n:
        plt.plot(np.correlate(n, n, mode="full"))

    plt.figure("hist")
    plt.hist(statemon_exc.n[0])


# Equations for the neurons and the synapses (below)
def eqs(sigma:str):
    """
    Equations for a rate model with signmoidal transfer function (dF/dt)
    syn_input as sum of exc. and inh. input.
    ext input as mean free noise
    """
    tau = ".1*ms"
    return f"""
        dn/dt = -n / ({tau}) + {sigma}*sqrt(2/({tau}))*xi_n : 1
        # dn/dt = {sigma}*sqrt(1/({tau}))*xi : 1
    """


if __name__ == "__main__":
    brian()
