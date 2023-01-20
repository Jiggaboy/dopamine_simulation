#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-30

@author: Hauke Wernecke

Controls all plots which are defined in the sub-modules.
"""

import cflogger
logger = cflogger.getLogger()

import numpy as np
import matplotlib.pyplot as plt

import lib.pickler as PIC
import universal as UNI


from custom_class import AngleDumper
from plot import Plot_ActivityDifference

from plot import ActivityDifferenceConfig as figcfg
from params import BaseConfig, TestConfig, PerlinConfig, NullConfig, ScaleupConfig

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


config = PerlinConfig()
center = config.center_range.values()
ANGLE_TAG = "angles_across_baselines_{}"

############################## START: CONTROL VARIABLES##############################
# Plot the differences in average activation across seeds in baseline simulations.
PLOT_BASELINE_ACROSS_SEEDS = True
# Plot the activity of a patchy simulation across the baseline of the same seed.
PLOT_ACTIVITY_DIFFERENCES = True
############################## END: CONTROL VARIABLES##############################


PATCH_AI_TAG = "alignment_index_{}"

ADD_INDIVIUAL_TRACES = False

def main():
    plot_activity_differences(PLOT_ACTIVITY_DIFFERENCES, PLOT_BASELINE_ACROSS_SEEDS)

    #for c in center:
        # angles_across_baselines(c, plot_traces=ADD_INDIVIUAL_TRACES)
    # for center_tag in config.center_range.keys():
    #     patch_ai(center_tag)


def plot_activity_differences(_activity_differences:bool=True, _baseline_across_seeds:bool=True):
    activity_difference = Plot_ActivityDifference(config, figcfg)
    if _baseline_across_seeds:
        activity_difference.baseline_difference_across_seeds()
    if _activity_differences:
        activity_difference.activity_difference()
    plt.show()


def patch_ai(center_tag:tuple):
    tag = PATCH_AI_TAG.format(center_tag)
    try:
        angle_dumper = PIC.load_angle_dumper(tag, sub_directory=config.sub_dir)
    except FileNotFoundError:
        logger.info(f"Skip: {tag} (file not found)")
        return


    figname = f"patch_ai_{center_tag}"
    fig, ax = plt.subplots(num=figname, figsize=(3, 3))


    title = f"Alignment Index (r={angle_dumper.radius[0]})"
    ax.set_title(title)
    ax.set_xlabel("PCs")
    ax.set_ylabel("Explained variance")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 3))
    hline = ax.axhline(.7, c="k", ls="dashed")

    var_bar = []
    for explained_variance in angle_dumper.explained_variances:
        bar = ax.errorbar(np.arange(1, explained_variance.shape[1]+1), explained_variance.mean(axis=0), yerr=explained_variance.std(axis=0))
        var_bar.append(bar)


    axis_ai = ax.twinx()
    axis_ai.set_ylabel('Alignment index')
    axis_ai.set_ylim(0, 1)
    axis_ai.set_yticks(np.linspace(0, 1, 3))
    indexes = angle_dumper.alignment_indexes
    ai_bar = axis_ai.errorbar(np.arange(1, indexes.shape[1]+1), indexes.mean(axis=0), yerr=indexes.std(axis=0), marker="*", c="green")

    # plt.legend([*var_bar, hline, ai_bar], ["Baseline", "w/ patch", "70% expl. variance", "Alignment Index"])
    plt.legend([*var_bar, ai_bar], ["Baseline", "w/ patch", "AI"], loc="lower right")
    plt.tight_layout()
    PIC.save_figure(figname, fig, config.sub_dir)



def angles_across_baselines(center:tuple, plot_traces:bool):
    tag = ANGLE_TAG.format(center)
    try:
        angle_dumper = PIC.load_angle_dumper(tag, sub_directory=config.sub_dir)
    except FileNotFoundError:
        logger.info(f"Skip: {tag} (file not found)")
        return

    figname = f"AVG_angles_{center}"
    fig, axes = plt.subplots(ncols=len(angle_dumper.radius), num=figname)

    for idx, (radius, angles) in enumerate(angle_dumper.angles.items()):
        ax = axes[idx]

        title = "Angles between PCs "
        title = title if radius is None else title + f"(r={radius})"
        ax.set_title(title)
        ax.set_xlabel("PCs")
        ax.set_ylabel("angle [°]")
        ax.set_ylim(0, 90)

        for x, (PC_angles, color) in enumerate(zip(angles, colors)):
            PC_angles = np.asarray(PC_angles)
            x_range = np.arange(1, x+2)
            ax.errorbar(x_range, PC_angles.mean(axis=0), yerr=PC_angles.std(axis=0), marker="_", lw=2, ms=10)

            if ADD_INDIVIUAL_TRACES:
                x_values = np.repeat(x_range[:, None], len(PC_angles), axis=1)
                ax.plot(x_values, PC_angles.T,  marker="o", ls="--", c=color, lw=1)
    PIC.save_figure(figname, fig, config.sub_dir)

if __name__ == "__main__":
    main()
    plt.show()
