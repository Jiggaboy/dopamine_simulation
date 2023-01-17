# -*- coding: utf-8 -*-
"""
This script runs several simulations defined by the configuration.
"""
import cflogger
log = cflogger.getLogger()

import matplotlib.pyplot as plt
import numpy as np

from custom_class.population import Population
import lib.pickler as PIC

import dopamine as DOP
import universal as UNI

from simulator import Simulator
#from NESTsimulator import NESTSimulator
from params import BaseConfig, TestConfig, PerlinConfig, StarterConfig, ScaleupConfig, NestConfig, LowDriveConfig
Config = TestConfig()
# Config = LowDriveConfig()
# Config = StarterConfig()

from lib import functimer


@functimer(logger=log)
def main():
    # Sets up a new population. Either loads the connectivity matrix or builds up a new one.
    neural_population = Population(Config)

    ## WARMUP
    simulator = Simulator(Config, neural_population)
    Config.save(subdir=simulator.sub_dir)
    # simulator.run_warmup()

    for seed in Config.drive.seeds:
        simulator.run_baseline(seed)

    return



def log_status(cfg:BaseConfig, radius, name, amount, percent):
    log.info("Simulation" \
          + f" radius: {cfg.RADIUSES.index(radius) + 1}/{len(cfg.RADIUSES)};"
          + f" name: {name};"
          + f" amount: {cfg.AMOUNT_NEURONS.index(amount) + 1}/{len(cfg.AMOUNT_NEURONS)};"
          + f" percent: {cfg.PERCENTAGES.index(percent) + 1}/{len(cfg.PERCENTAGES)};")



if __name__ == "__main__":
    main()
    plt.show()


    # tau = 12.
    # Config.transfer_function.offset = -0
    # W = np.asarray([[0, 0, -100], [-100, 0., 0], [0, -100, 0]]).T
    # def exp_decay_of_TF_with_GWN(t, y, tau):
    #     return 1/tau * (-y + Config.transfer_function.run(W.dot(y) + gwn[int(t)]))

    # from scipy.integrate import solve_ivp
    # T_END = 100
    # t_axis = np.arange(0, T_END, .1)

    # gwn = np.random.normal(scale=1., size=(T_END, 3))

    # sol = solve_ivp(exp_decay_of_TF_with_GWN, [t_axis[0], t_axis[-1]], [.5, .6, .4], t_eval=t_axis, args=[tau])

    # for y in sol.y:
    #     plt.plot(sol.t, y)
    # print(sol.status)
