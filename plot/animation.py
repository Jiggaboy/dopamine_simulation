#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Animation of activity data.

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

from dataclasses import dataclass
from functools import partial
import numpy as np
import matplotlib.pyplot as plt


from lib import pickler as PIC
import lib.universal as UNI
from plot.activity import animate_firing_rates, create_image, get_width
from plot import COLOR_MAP_DIFFERENCE, NORM_DIFFERENCE, COLOR_MAP_ACTIVITY, NORM_ACTIVITY
from plot import AnimationConfig as figcfg
from plot.lib import add_colorbar



BASELINES = True
PATCHES = False
BASELINE_DIFFERENCES = False


def animate(config:object, animate_baseline, animate_patch, animate_baseline_differences:bool=False):

    animator = Animator(config, figcfg)
    if animate_baseline:
        animator.animate(config.baseline_tags[:1])
    if animate_patch:
        animator.animate(config.get_all_tags(seeds=0))
    if animate_baseline_differences:
        animator.baseline_difference_animations()
    animator.show()


def main():
    from params import SelectConfig, config
    # config = SelectConfig()

    animator = Animator(config, figcfg)
    if BASELINES:
        animator.animate(config.baseline_tags[:1])
        # animator.animate_spikes(config.baseline_tags)
    if PATCHES:
        animator.animate(config.get_all_tags(seeds=0))
    if BASELINE_DIFFERENCES:
        animator.baseline_difference_animations()
    animator.show()


@dataclass
class Animator:
    config: object
    fig_config: object

    def __post_init__(self):
        self.coordinates = UNI.get_coordinates(nrows=self.config.rows, step=1)
        self.animations = []


    @property
    def width(self):
        return self.config.rows


    def show(self):
        """Maps hte matplorlib.pyplot.show to the class."""
        plt.show()


    ########################################################################################################
    def update_spikes(self, i:int, spikes:np.ndarray, step:int=None, axis:object=None, line:object=None, **kwargs):
        time, coordinates = spikes[:, 0], spikes[:, 1:]
        if line is None:
            return axis.plot(*coordinates[0], color="lime", marker=".", ls="none")[0]

        idx = np.where(np.logical_and(i-step < time, time < i))[0]
        return line.set_data(*coordinates[idx].T)

    def animate_spikes(self, tag:list, axis:object, fig:object)->None:
        import analysis.dbscan_sequences as dbs
        scanner = dbs.DBScan_Sequences(self.config)
        spikes, _ = scanner._scan_spike_train(tag)
        if spikes.size == 0:
            logger.error("No spike detected.")
        line = self.update_spikes(spikes=spikes, i=0, axis=axis)

        method = partial(self.update_spikes, spikes=spikes, step=self.fig_config.animation_kwargs["step"], line=line)
        self._set_stop(spikes[:, 0].max())
        anim = animate_firing_rates(fig, method, **self.fig_config.animation_kwargs)
        self.animations.append(anim)

    ########################################################################################################


    def animate(self, tags:list)->None:
        for tag in tags:
            logger.info(f"Animate baseline tag: {tag}")
            rate = self._load_rate(tag)
            self.baseline_figure(tag, rate)


    def baseline_difference_animations(self)->list:
        """Creates a plot across all baseline simulations and animates along time."""
        cross_differences = self._load_rate_differences(self.config.baseline_tags)
        no_of_seeds = len(self.config.simulation_seeds)
        fig, axes = plt.subplots(nrows=no_of_seeds, ncols=no_of_seeds, **self.fig_config.difference_frame)
        fig.suptitle("Difference of neuronal activity over time")
        self.baseline_difference_figure(cross_differences, axes=axes, fig=fig)


    def baseline_figure(self, tag:str, rate:np.ndarray):
        fig, axis = plt.subplots(num=f"activity_{tag}", **self.fig_config.figure_frame)
        fig.suptitle("Neuronal activity evolves over time")

        cbar = add_colorbar(axis, **self.fig_config.image)
        cbar.ax.get_yaxis().labelpad = 15
        # cbar.ax.set_yticklabels(['low','med.','high'])
        axis.set_xticks([0, 40, 80])
        axis.set_yticks([0, 40, 80])
        cbar.set_label('activation [a.u.]', rotation=0)
        if self.fig_config.animation_kwargs.get("add_spikes", False):
            self.animate_spikes(tag, axis, fig)

        image = update_activity_plot(rate=rate.T, i=self.fig_config.animation_kwargs.start, axis=axis, **self.fig_config.image)
        method = partial(update_activity_plot, im=image, rate=rate.T, axis=axis)
        self._set_stop(rate.shape[-1])
        anim = animate_firing_rates(fig, method, **self.fig_config.animation_kwargs)
        self.animations.append(anim)
        self.save_animation(fig, anim)


    def baseline_difference_figure(self, rate:np.ndarray, axes:object, fig:object)->None:
        ims = init_activity_plots(rate, i=self.fig_config.animation_kwargs.start, axes=axes, **self.fig_config.difference_image)
        method = partial(update_activity_plots, rate, axes=axes, images=ims, **self.fig_config.difference_image)
        for axis in axes:
            cbr = add_colorbar(axis, **self.fig_config.difference_image)

        self._set_stop(rate.shape[-1])
        anim = animate_firing_rates(fig, method, **self.fig_config.animation_kwargs)
        self.animations.append(anim)
        self.save_animation(fig, anim)


    def _set_stop(self, t_end:int)->None:
        if self.fig_config.animation_kwargs.stop == None:
            self.fig_config.animation_kwargs.stop = t_end


    def save_animation(self, fig:object, anim:object):
        if self.fig_config.save_animations:
            fig.tight_layout()
            plt.show()
            from lib.pickler_class import Pickler
            pickler = Pickler(self.config)
            pickler.save_animation(fig.get_label(), anim)


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


def update_activity_plot(i:int, axis:object, rate:np.ndarray, im:object=None, **kwargs):
    axis.set_title(f"Activity at point t = {i}")
    if im is None:
        return create_image(rate[i], axis=axis, **kwargs)
    width = get_width(rate[i].size)
    return im.set_array(rate[i].reshape((width, width)))


def init_activity_plots(rate:np.ndarray, i:int, axes:object, **kwargs)->np.ndarray:
    ims = np.empty(shape=rate.shape[:2], dtype=object)
    for row in range(rate.shape[0]):
        for col in range(rate.shape[1]):
            axes[row, col].set_title(f"Activity at point t = {i}")
            ims[row, col] = create_image(rate[row, col, :, i], axis=axes[row, col], **kwargs)
    return ims

def update_activity_plots(rate:np.ndarray, i:int, axes:object, images:list, **kwargs):
    for row in range(rate.shape[0]):
        for col in range(rate.shape[1]):
            if row == col:
                continue
            axes[row, col].set_title(f"Activity at point t = {i}")
            width = get_width(rate[row, col, :, i].size)
            images[row, col].set_array(rate[row, col, :, i].reshape((width, width)))


if __name__ == "__main__":
    main()
