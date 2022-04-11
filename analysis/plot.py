#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:01:17 2021

@author: hauke
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable

import configuration as CF
import dopamine as DOP

import custom_class.pickler as PIC

import animation.activity as ACT
import animation.rate as RAT
import animation.misc as misc

import universal as UNI

keys, space = UNI.get_parameter_space()
print(keys, space)

MODE = "Perlin_uniform"
bs_tag = UNI.get_tag_ident(MODE, "baseline")
c_idx = list(keys).index("center")
r_idx = list(keys).index("radius")

def plot():
    plot_baseline_activity(bs_tag)


    for p in space:
        tags = UNI.get_tag_ident(MODE, *p)

        try:
            c = CF.center_range[p[c_idx]]
            r = float(p[r_idx])
            plot_rates_vs_baseline(tags, bs_tag, center=c, radius=r)
        except FileNotFoundError:
            print(tags)
            break


def plot_baseline_activity(bs_tag, save:bool=True):
    bs_rate = PIC.load_rate(bs_tag, skip_warmup=True, exc_only=True)
    bs_act = bs_rate.mean(axis=1)

    figname = f"{bs_tag}"
    title = f"Average activity: {100 * bs_act.mean():.2f}%"
    ACT.activity(bs_act, figname=figname, title=title, figsize=(12, 8), norm=(0., 0.5))
    if save:
        plt.savefig(UNI.get_fig_filename(figname))



def plot_rates_vs_baseline(postfixes:list, baseline:str=None, save:bool=False, **kwargs):
    if isinstance(postfixes, str):
        postfixes = (postfixes, )
    rates = merge_avg_rate_to_key(postfixes, **kwargs)
    if baseline is not None:
        bs = PIC.load_rate(baseline, skip_warmup=True, exc_only=True)
        baseline_rate = bs.mean(axis=1)
        for rate in rates.items():
            plot_rate_difference(rate, baseline_rate, norm=(-.3, .3))
            center = kwargs.get("center")
            if center is not None:
                radius = kwargs.get("radius")
                misc.plot_patch(center, radius)
            if save:
                plt.savefig(UNI.get_fig_filename(rate[0]))


def merge_avg_rate_to_key(keys:list, plot:bool=False, center:tuple=None, radius:float=4, title:str=None)->dict:
            rates = {}
            for s in keys:
                rate = PIC.load_rate(s, skip_warmup=True, exc_only=True)
                avgRate = rate.mean(axis=1)
                rates[s] = avgRate
                # if plot:
                #     title= title or "Activity averaged across time"
                #     ACT.activity(avgRate, title=title, figname=f"circ_patch_{s}", norm=(0, 0.5))
                #     if center is not None:
                #         plot_patch(center, radius)
            return rates


def plot_rate_difference(avg_rate:(str, np.ndarray), baseline:np.ndarray, norm:tuple=None):
    norm = norm or (None, None)

    rate_diff = avg_rate[1] - baseline
    diff_percent = rate_diff.mean() / baseline.mean()
    # To be adjusted
    figname = f"circ_patch_{avg_rate[0]}_bs"
    title = f"Network changes: \nActivation difference: {100 * diff_percent:+.2f}%"
    ACT.activity(rate_diff, figname=figname, title=title, norm=norm, cmap=plt.cm.seismic, figsize=(12, 8))



if __name__ == "__main__":
    plot()
