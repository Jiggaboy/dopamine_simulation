#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""

from cflogger import logger


from dataclasses import dataclass, field
import numpy as np
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt

import lib.pickler as PIC
from lib.pickler_class import Pickler
import lib.universal as UNI

from plot.lib import image_slider_2d, image_slider_1d, plot_patch
from plot.lib.frame import create_image
from plot import activity
from plot import ActivityDifferenceConfig as figcfg

from params import config


def main():
    plot_activity_differences(config, patch_vs_baseline=True, baseline_across_seeds=True)


def plot_activity_differences(config:object, patch_vs_baseline:bool, baseline_across_seeds:bool):
    activity_difference = Plot_ActivityDifference(config, figcfg)
    if patch_vs_baseline:
        tags = config.get_all_tags(seeds="all")
        activity_difference.activity_difference(tags)
    # if baseline_across_seeds:
    #     activity_difference.baseline_difference_across_seeds()


@dataclass
class Plot_ActivityDifference:
    _config: object
    _fig_config: object
    _slider: list = field(default_factory=list)

    def activity_difference(self, tags:list, **kwargs):
        for tag in tags:
            # {tag} is a list of tags (grouped by seeds)
            logger.info(f"Plot activity difference for tag: {tag}")
            self._patch_vs_baseline(tag)


    def _patch_vs_baseline(self, tag:str)->None:
        # pooled rates: seed specific differences
        pooled_rates = self._rate_differences_against_baseline(tag)
        # Slider Plot: Patch vs bs per seed
        self._create_patch_difference_plot(tag, pooled_rates.T)
        # Patch vs BS (averaged)
        self._create_patch_average_difference_plot(tag, pooled_rates)
        pass

    def baseline_difference_across_seeds(self)->None:
        """
        Calculates the avg. differences between all simulations given by tags.
        Create a plot with sliders to see the differences.
        """
        figname = "Differences in baseline conditions"
        title = "Differences in baseline conditions"
        fig, axes = self._frame(figname, title)

        pooled_diffs = self._rate_differences(self._config.baseline_tags)

        slider = image_slider_2d(pooled_diffs, fig, axis=axes, label="seed", **figcfg.image)
        self._slider.append(slider)
        return fig, slider


    def _create_patch_difference_plot(self, tag:str, data:np.ndarray):
        full_name, _ = UNI.split_seed_from_tag(tag[0])
        figname = f"Average_diff_patch_{full_name}"
        title = "Differences in patch against baseline simulation"
        # fig, axes = self._frame(figname, title)
        slide_label = "Seed"

        fig, axes = plt.subplots(ncols=len(data), num=figname)
        fig.suptitle(title)
        print(data.shape)
        for ax, d in zip(axes, data):
            create_image(d, axis=ax, **figcfg.image)
            plt.sca(ax)
            plot_patch_from_tag(tag[0], config)


        # # prepare the method that is called when the slider is moved.
        # method = partial(self.update_patch_difference, data=data, fig=fig, axis=axes, tag=tag[0], config=self._config)
        # s = image_slider_1d(data, fig, axis=axes, label=slide_label, method=method)
        # self._slider.append(s)
        # return s


    def _rate_differences(self, tags:list)->np.ndarray:
        """
        Calculates the avg. differences between all simulations given by tags.
        """
        tags = UNI.make_iterable(tags)
        pooled_diffs = np.zeros((len(tags), len(tags), self._config.no_exc_neurons), dtype=float)

        for i, tag1 in enumerate(tags):
            avg_rate1 = PIC.load_average_rate(tag1, sub_directory=self._config.sub_dir, config=self._config)
            for j, tag2 in enumerate(tags):
                avg_rate2 = PIC.load_average_rate(tag2, sub_directory=self._config.sub_dir, config=self._config)
                rate_diff = avg_rate1 - avg_rate2
                pooled_diffs[i, j] = rate_diff
        return np.asarray(pooled_diffs)



    def _rate_differences_against_baseline(self, tags:list)->np.ndarray:
        """
        Calculates the avg difference against baseline for all seeds.

        Parameters
        ----------
        tags : list
            DESCRIPTION.

        Returns
        -------
        np.ndarray
            Pooles the differences of rates against the baseline.

        """
        pooled_rates = pd.DataFrame()

        for t in tags:
            _, seed = UNI.split_seed_from_tag(t)
            baseRate = PIC.load_average_rate(self._config.baseline_tag(seed), sub_directory=self._config.sub_dir, config=self._config)
            avgRate = PIC.load_average_rate(t, sub_directory=self._config.sub_dir, config=self._config)
            if baseRate is None or avgRate is None:
                logger.warning(f"Could not find averaged baseline or patch rates ({t}).")
                continue

            rate_diff = avgRate - baseRate
            tmp = pd.DataFrame(rate_diff, columns=[t])
            pooled_rates = pd.concat([pooled_rates, tmp], axis=1)
        return np.asarray(pooled_rates)


    def _create_patch_average_difference_plot(self, tag:list, rates:np.ndarray):
        """

        """
        full_name, _ = UNI.split_seed_from_tag(tag[0])
        title = f"Avg. activation difference: {100 * rates.mean():+.2f}%"
        fig = activity.activity(rates.mean(axis=1), figname=full_name, title=title, **figcfg.image, **figcfg.figure_frame)
        plot_patch_from_tag(tag[0], self._config)
        pickler = Pickler(self._config)
        pickler.save_figure(full_name, fig)


    def _frame(self, figname:str, title:str):
        fig, axes = plt.subplots(num=figname, **self._fig_config.figure_frame)
        fig.suptitle(title, **self._fig_config.font)
        return fig, axes


    @staticmethod
    def update_patch_difference(data:np.ndarray, fig, axis, tag:str, config:object, idx:int):
        create_image(data[idx], axis=axis, **figcfg.image)
        axis.set_title(f"Specifier: {tag} (seed: {idx})")
        plot_patch_from_tag(tag, config)



def plot_patch_from_tag(tag:str, config:object):
    name = UNI.name_from_tag(tag)
    center = config.get_center(name)

    radius =  UNI.radius_from_tag(tag)
    if np.asarray(center).size > 2:
        for c in center:
            plot_patch(c, float(radius), width=config.rows)
    else:
        plot_patch(center, float(radius), width=config.rows)


if __name__ == "__main__":
    main()
    plt.show()
