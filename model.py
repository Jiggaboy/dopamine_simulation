# -*- coding: utf-8 -*-
"""
This script runs several simulations defined by the configuration.
"""
import cflogger
log = cflogger.getLogger()

import matplotlib.pyplot as plt
import numpy as np

from custom_class.population import Population
import util.pickler as PIC

import dopamine as DOP
import universal as UNI

from simulator import Simulator
from params import BaseConfig, TestConfig, PerlinConfig, StarterConfig, ScaleupConfig
#Config = TestConfig()
Config = PerlinConfig()
#Config = ScaleupConfig()

from util import functimer


#%% Intialisation


@functimer(logger=log)
def main():
    # Sets up a new population. Either loads the connectivity matrix or builds up a new one.
    neural_population = Population(Config)

    ## WARMUP
    simulator = Simulator(Config, neural_population)
    Config.save(subdir=simulator.sub_dir)
    simulator.run_warmup()

    for seed in Config.drive.seeds:
        simulator.run_baseline(seed)
        for radius in Config.RADIUSES[:]:
            for name, center in Config.center_range.items():
                # Create Patch and retrieve possible affected neurons
                dop_area = DOP.circular_patch(Config.rows, center, radius)
                for amount in Config.AMOUNT_NEURONS[:]:
                    # Select affected neurons
                    dop_patch = np.random.choice(dop_area.nonzero()[0], amount, replace=False)
                    for percent in Config.PERCENTAGES[:]:
                        log_status(Config, radius=radius, name=name, amount=amount, percent=percent)

                        tag = UNI.get_tag_ident(name, radius, amount, int(percent*100), seed)
                        log.info(f"Current tag: {tag}")

                        simulator.run_patch(dop_patch, percent, tag, seed)

    #avg_baseline_activity(Config)
    # Load last simulation to get an impression of the activity
    anim = animate_firing_rates(Config.baseline_tags[0], neural_population, simulator)
    anim2 = animate_firing_rates(Config.baseline_tags[1], neural_population, simulator)
    anim3 = animate_firing_rates(Config.baseline_tags[2], neural_population, simulator)
    plt.show()
    

def log_status(cfg:BaseConfig, radius, name, amount, percent):
    log.info("Simulation" \
          + f" radius: {cfg.RADIUSES.index(radius) + 1}/{len(cfg.RADIUSES)};"
          + f" name: {name};"
          + f" amount: {cfg.AMOUNT_NEURONS.index(amount) + 1}/{len(cfg.AMOUNT_NEURONS)};"
          + f" percent: {cfg.PERCENTAGES.index(percent) + 1}/{len(cfg.PERCENTAGES)};")


def avg_baseline_activity(cfg):
    from plot import avg_activity
    for baseline_tag in cfg.baseline_tags:
        avg_activity(baseline_tag, config=cfg)
    #from analysis.plot import plot_baseline_activity
    #plot_baseline_activity(cfg.baseline_tag, config=cfg)
    
    
def animate_firing_rates(tag:str, neural_population:Population, simulator):
    """Loads the rate of _tag_ and animate the activity."""
    rate = simulator._load_rate(tag)
    from animation.activity import animate_firing_rates
    return animate_firing_rates(rate, neural_population.coordinates, neural_population.exc_neurons.size, fig_tag=tag, start=500, interval=100)
    


if __name__ == "__main__":
    main()
    plt.show()
    quit()