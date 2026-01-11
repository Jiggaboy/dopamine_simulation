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
        H, edges = np.histogram(rate.ravel(), density=True, bins=bins)
        bs_rates[t] = H
        
        portion = H[edges[:-1] >= RATE_THRESHOLD].sum() / H.sum()
        portions[t] = portion
    
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    plt.yscale = "log"
    plt.bar(bin_centers, bs_rates.mean(axis=0), yerr=bs_rates.std(axis=0)*100, width=0.05)
    plt.text(x=0.5, y=bs_rates.mean(), s=f"p={portions.mean():.4f}")
    print(portions)

    
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
