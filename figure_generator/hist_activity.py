#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Making a histogram of the rates in the baseline simulations.
    Maybe help to find proper values for the threshold of sequence detection.

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lib import pickler as PIC
from params import config

N_rate = 15
RATE_THRESHOLD = .5
N_syn = 23
SYN_THRESHOLD = 20

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

def main():
    bs_rates = np.zeros((len(config.baseline_tags), N_rate-1))
    portions = np.zeros(len(config.baseline_tags))
    
    bins = np.linspace(0, 1, N_rate, endpoint=True)    
    for t, tag in enumerate(config.baseline_tags):
        rate = PIC.load_rate(tag, skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)
        H, edges = np.histogram(rate.ravel(), density=True, bins=bins)
        bs_rates[t] = H
        
        portion = H[edges[:-1] >= RATE_THRESHOLD].sum() / H.sum()
        portions[t] = portion
    
    
    gs_kw = dict(width_ratios=[3, 1, 1])
    # fig, (ax_rate, ax_hist) = plt.subplots(ncols=3, figsize=(8, 2), gridspec_kw=gs_kw, layout="constrained")
    fig = plt.figure(figsize=(8, 2), layout="constrained")
    left, right = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[3, 1], figure=fig)
    ax_rate = fig.add_subplot(left)

    inner = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=right, width_ratios=[1.4, 1], wspace=0)
    ax_hist_low  = fig.add_subplot(inner[0], sharey=ax_rate)
    ax_hist_high = fig.add_subplot(inner[1], sharey=ax_hist_low)
      
    ax_rate.set_xlabel("time [ms]")
    ax_rate.set_ylabel("rate [au]")
    ax_rate.set_xlim(low, high)
    
    ax_hist_low.set_xlabel("              probability?")
    # ax_hist_low.set_ylabel("rate [au]")
    ax_hist_low.set_xlim(0, .8)
    ax_hist_high.set_xlim(10, 13)
    
    # Hide spines between axes
    ax_hist_low.spines["right"].set_visible(False)
    ax_hist_high.spines["left"].set_visible(False)
    ax_hist_low.tick_params(labelright=False)
    ax_hist_low.tick_params(labelleft=False)
    ax_hist_high.tick_params(labelleft=False)
    ax_hist_high.tick_params(
        axis='y',          
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off 
        direction="in",
        length=0,
        width=0,
        pad=0,
        labelsize=0,
    )
    # ax_hist_high.yaxis.set_ticks_position('none')
    # ax_hist_high.set('none')
    
    # Diagonal break marks
    d = 0.02
    kwargs = dict(transform=ax_hist_low.transAxes, color="k", clip_on=False)
    ax_hist_low.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax_hist_low.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs = dict(transform=ax_hist_high.transAxes, color="k", clip_on=False)
    ax_hist_high.plot((-d, +d), (-d, +d), **kwargs)
    ax_hist_high.plot((-d, +d), (1-d, 1+d), **kwargs)
    
    handles = []
    for i in [150, 298, 3459, 5330, 5928]:
        handle = ax_rate.plot(np.arange(low, high), rate[i][low:high], label=f"#{i:4}")
        handles.append(handle[0])
    ax_rate.legend(handles=handles, handlelength=0, handletextpad=0, prop={'family': 'monospace'})
    # neurons = np.random.choice(np.arange(6400), 6, replace=False)
    # print(neurons)
    # for i in neurons:
    #     ax_rate.plot(rate[i])

    bin_centers = 0.5*(bins[1:]+bins[:-1])
    ax_hist_low.barh(bin_centers, bs_rates.mean(axis=0), xerr=bs_rates.std(axis=0), height=0.05)
    ax_hist_high.barh(bin_centers, bs_rates.mean(axis=0), xerr=bs_rates.std(axis=0), height=0.05)
    
    ax_hist_low.axhline(config.analysis.sequence.spike_threshold, ls="--", c="k")


def hist_activity():
    bs_rates = np.zeros((len(config.baseline_tags), N_rate-1))
    portions = np.zeros(len(config.baseline_tags))
    bins = np.linspace(0, 1, N_rate, endpoint=True)    
    for t, tag in enumerate(config.baseline_tags):
        H, edges = np.histogram(rate.ravel(), density=True, bins=bins)
        bs_rates[t] = H
        
        portion = H[edges[:-1] >= RATE_THRESHOLD].sum() / H.sum()
        portions[t] = portion
    
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    plt.yscale = "log"
    plt.bar(bin_centers, bs_rates.mean(axis=0), yerr=bs_rates.std(axis=0)*100, width=0.05)
    plt.text(x=0.5, y=bs_rates.mean(), s=f"p={portions.mean():.4f}")
    print(portions)


def hist_synaptic_input():
    plt.figure()
    for tag in config.baseline_tags:
        try:
            synaptic_input = PIC.load_synaptic_input(tag, sub_directory=config.sub_dir)
        except FileNotFoundError:
            print("No syn. input found...")
            break
        bins = np.linspace(-225, 325, N_syn, endpoint=True)
        H, edges = np.histogram(synaptic_input.ravel(), density=True, bins=bins)
        portion = H[edges[:-1] >= SYN_THRESHOLD].sum() / H.sum()
        plt.step(edges[:-1], H)
        plt.text(x=0, y=H.mean(), s=f"p={portion:.4f}")
        print(synaptic_input.min(), synaptic_input.max())
        
        # ext. input
        # We can just take the std here, as the noise is defined by n_dot = -n/tau + sigma*sqrt(2/tau)*GWN
        # And for such an OU process, the std in stationarity is defined as sqrt(tau/2)*std of the OU process which has the inverse prefactor.
        drive = np.random.normal(config.drive.mean, config.drive.std, size=1000) 
        plt.hist(drive, bins=bins, density=True)
        break
    
    plt.axvline(config.transfer_function.offset, ls="--", c="k")

#===============================================================================
# METHODS
#===============================================================================



if __name__ == '__main__':
    main()
    plt.show()
