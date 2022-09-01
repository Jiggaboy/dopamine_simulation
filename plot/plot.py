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

import util.pickler as PIC
import universal as UNI


from custom_class import AngleDumper
    
    
from params import BaseConfig, TestConfig, PerlinConfig, NullConfig, ScaleupConfig

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


config = PerlinConfig()
center = config.center_range.values()
ANGLE_TAG = "angles_across_baselines_{}"

ADD_INDIVIUAL_TRACES = False

def main():
    for c in center:
        angles_across_baselines(c, plot_traces=ADD_INDIVIUAL_TRACES)


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