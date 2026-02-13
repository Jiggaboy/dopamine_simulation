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
import lib.universal as UNI

from plot.lib.frame import create_image
from plot.lib.basic import add_colorbar, plot_patch_from_tag, plot_patch
from plot.constants import NORM_DIFFERENCE, COLOR_MAP_DIFFERENCE

from params import config

image_config = {
    "norm": (-.25, .25),
    "cmap": plt.cm.seismic,
}


@dataclass
class Plot_ActivityDifference:
    _config: object
    _slider: list = field(default_factory=list)

    # TODO: REFACTOR!
    def activity_difference(self, tags:list, **kwargs):
        for tag in tags:
            # {tag} is a list of tags (grouped by seeds)
            logger.info(f"Plot activity difference for tag: {tag}")
            self._patch_vs_baseline(tag)


    def _patch_vs_baseline(self, tag:str)->None:
        # pooled rates: seed specific differences
        pooled_rates = self._rate_differences_against_baseline(tag)
        self._create_patch_difference_plot(tag, pooled_rates.T)
        # Patch vs BS (averaged)
        self._create_patch_average_difference_plot(tag, pooled_rates)


    def _create_patch_difference_plot(self, tag:str, data:np.ndarray):
        full_name, _ = UNI.split_seed_from_tag(tag[0])
        figname = f"Average_diff_patch_{full_name}"
        title = "Differences in patch against baseline simulation"

        fig, axes = plt.subplots(ncols=len(data), num=figname)
        fig.suptitle(title)
        for ax, d in zip(axes, data):
            create_image(d, axis=ax, **image_config)
            plt.sca(ax)
            plot_patch_from_tag(tag[0], config)



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

        Returns
        -------
        np.ndarray
            Pooles the differences of rates against the baseline.

        """
        from lib.neuralhdf5 import NeuralHdf5, default_filename
        
        pooled_rates = pd.DataFrame()

        with NeuralHdf5(default_filename, "r", config=self._config) as file:
            for t in tags:
                # _, seed = UNI.split_seed_from_tag(t)
                # baseRate = PIC.load_average_rate(self._config.baseline_tag(seed), sub_directory=self._config.sub_dir, config=self._config)
                # avgRate = PIC.load_average_rate(t, sub_directory=self._config.sub_dir, config=self._config)
                baseRate = file.get_average_rate(self._config.get_baseline_tag_from_tag(t), is_baseline=True)
                avgRate  = file.get_average_rate(t)
                
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
        name = UNI.name_from_tag(tag[0])
        fig, ax = plt.subplots(num=full_name, figsize=(5.5, 3.5))
        plt.sca(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks([10, 40, 70])
        ax.set_yticks([10, 40, 70])
        # fig.suptitle(f"Avg. activation difference: {100 * rates.mean():+.2f}%")
        norm = NORM_DIFFERENCE
        cmap = COLOR_MAP_DIFFERENCE

        plot_patch_from_tag(tag[0], self._config)

        for ds in self._config.analysis.dbscan_controls.detection_spots:
            ds_name, spots = ds
            if ds_name == name:
                for spot in spots:
                    plot_patch(spot, radius=2., width=self._config.rows, axis=ax, ec="lime", ls="solid")


        im = create_image(rates.mean(axis=1), norm, cmap, axis=ax)
        cbar = add_colorbar(ax, norm, cmap)
        cbar.set_label("Avg. activity [a.u.]", rotation=270, labelpad=15)


        # plt.tight_layout()
        PIC.save_figure(full_name, fig, sub_directory=self._config.sub_dir, transparent=True)




    @staticmethod
    def update_patch_difference(data:np.ndarray, fig, axis, tag:str, config:object, idx:int):
        create_image(data[idx], axis=axis, **image_config)
        axis.set_title(f"Specifier: {tag} (seed: {idx})")
        plot_patch_from_tag(tag, config)
