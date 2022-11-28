#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-25

@author: Hauke Wernecke


Here we animate the simulation data.

"""

# Start with one specific one
import cflogger
logger = cflogger.getLogger()

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from functools import partial

from params import PerlinConfig

from custom_class.population import Population
from lib import pickler as PIC
import universal as UNI
from animation.activity import animate_firing_rates, create_image, get_width
from plot import COLOR_MAP_DIFFERENCE, NORM_DIFFERENCE, COLOR_MAP_ACTIVITY, NORM_ACTIVITY
from plot.lib import add_colorbar, plot_patch

START = 950
STOP = 1050
FPS = 2

SAVE_ANIMATIONS = False

BASELINES = True
BASELINE_DIFFERENCES = False

BS_FIGSIZE = (8, 6)

"""
    import matplotlib.animation as animation
    writergif = animation.PillowWriter(fps=30)
"""

def main():
    config = PerlinConfig()

    animator = Animator(config)
    if BASELINES:
        animator.baseline_animations()
    if BASELINE_DIFFERENCES:
        animator.baseline_difference_animations()

    animator.show()


def update_activity_plot(rate:np.ndarray, i:int, axis:object, **kwargs):
    axis.set_title(f"Activity at point t = {i}")
    return create_image(rate[i], axis=axis, **kwargs)


def update_activity_plot2(i:int, im:object, axis:object, rate:np.ndarray, **kwargs):
    axis.set_title(f"Activity at point t = {i}")
    width = get_width(rate[i].size)
    return im.set_array(rate[i].reshape((width, width)))
    return im.set_array(rate[i].reshape((width, width)))


def update_activity_plots(rate:np.ndarray, i:int, axes:object, **kwargs):
    for row in range(rate.shape[0]):
        for col in range(rate.shape[1]):
            axes[row, col].set_title(f"Activity at point t = {i}", fontsize=8)
            create_image(rate[row, col, :, i], axis=axes[row, col], **kwargs)


class Animator:

    def __init__(self, config:object):
        self.config = config
        self.coordinates = Population.populate_subgrid(config.rows, config.rows, step=1)
        self.animations = []


    def show(self):
        plt.show()


    def baseline_animations(self)->list:
        for bs_tag in self.config.baseline_tags[:1]:
            logger.info(f"Animate baseline tag: {bs_tag}")
            bs_rate = self._load_rate(bs_tag)
            self.baseline_figure(bs_tag, bs_rate)


    def baseline_difference_animations(self)->list:
        cross_differences = self._load_rate_differences(self.config.baseline_tags)
        no_of_seeds = len(self.config.simulation_seeds)
        figkwargs = {
            "num": "baseline_differences",
            "figsize": (16, 14)
        }
        fig, axes = plt.subplots(nrows=no_of_seeds, ncols=no_of_seeds, **figkwargs)
        fig.suptitle("Difference of neuronal activity over time")

        self.baseline_difference_figure(cross_differences, axes=axes, fig=fig)


    def baseline_figure(self, tag:str, rate:np.ndarray):
        fig, axis = plt.subplots(num=f"activity_{tag}", figsize=BS_FIGSIZE)
        fig.suptitle("Neuronal activity evolves over time")

        cmap = COLOR_MAP_ACTIVITY
        norm = NORM_ACTIVITY

        method = partial(update_activity_plot, rate.T, axis=axis, cmap=cmap, norm=norm)
        add_colorbar(axis, norm, cmap)
        plot_patch((28, 26), 2, self.config.rows)
        plot_patch((30, 18), 2, self.config.rows)

        image = method(i=1)

        method = partial(update_activity_plot2, im=image, rate=rate.T, axis=axis)

        anim = animate_firing_rates(fig, method, start=START, interval=1000 / FPS, stop=STOP, step=2)
        self.animations.append(anim)
        if SAVE_ANIMATIONS:
            PIC.save_animation(tag, anim, self.config.sub_dir)


    def baseline_difference_figure(self, rate:np.ndarray, axes:object, fig:object)->None:
        cmap = COLOR_MAP_DIFFERENCE
        norm = NORM_DIFFERENCE

        method = partial(update_activity_plots, rate, axes=axes, cmap=cmap, norm=norm)
        for axis in axes:
            add_colorbar(axis, norm, cmap)

        anim = animate_firing_rates(fig, method, start=START, interval=1000 / FPS, stop=rate.shape[-1])
        self.animations.append(anim)
        if SAVE_ANIMATIONS:
            PIC.save_animation(fig.get_label(), anim, self.config.sub_dir)


    def _load_rate(self, tag:str):
        return PIC.load_rate(tag, skip_warmup=True, exc_only=True, sub_directory=self.config.sub_dir, config=self.config)


    def _load_rate_differences(self, tags:list)->np.ndarray:
        """
        Calculates the avg. differences between all simulations given by tags.
        """
        tags = UNI.make_iterable(tags)
        pooled_diffs = np.zeros((len(tags), len(tags), self.config.no_exc_neurons, self.config.sim_time), dtype=float)

        for i, tag1 in enumerate(tags):
            rate1 = self._load_rate(tag1)
            for j, tag2 in enumerate(tags):
                rate2 = self._load_rate(tag2)
                rate_diff = rate1 - rate2
                pooled_diffs[i, j] = rate_diff
        return np.asarray(pooled_diffs)


if __name__ == "__main__":
    main()
