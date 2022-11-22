#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:37:12 2021

@author: hauke
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import cflogger
cflogger.set_up()
log = logging.getLogger()

import lib.pickler as PIC
import dopamine as DOP
import universal as UNI

import peakutils as putils

from lib import SequenceCounter


### Idea: Plot the number of sequences passed through a patch

# TODO: Add some discrimination line?
# TODO: Reduce the number of data points to a specific parameter of interest.
# TODO: Uniform scale


# MODE = "Perlin_uniform"
# bs_tag = UNI.get_tag_ident(MODE, "baseline")

# center_pre = (21, 65)
# center_post = (30, 63)
# center_3 = (23, 44)
# center = (center_pre, center_post, center_3, )
# radius = 2

def main():
    from params import PerlinConfig
    cf = PerlinConfig()

    all_tags = cf.get_all_tags()
    log.info(f"Analysis of passing sequences: Tags are: {all_tags}")
    log.info(f"Analysis of passing sequences: Center names are: {cf.center_range.keys()}")

    R = 2

    patches = []


    # name = "in"
    # center = ((30, 18), (28, 26), )
    # patches.append((name, center))
    # name = "edge"
    # patches.append((name, center))
    # name = "out"
    # patches.append((name, center))
    #
    # name = "edge-activator"
    # center = ((35, 49), (49, 36), )
    # patches.append((name, center))
    # name = "in-activator"
    # patches.append((name, center))
    # name = "out-activator"
    # patches.append((name, center))
    #
    # name = "linker"
    # center = ((21, 65), (30, 61), )
    # patches.append((name, center))
    #
    # name = "repeater"
    # center = ((2, 31), (29, 35), (29, 25))
    # patches.append((name, center))

    name = "starter"
    center = ((47, 4), (48, 8))
    patches.append((name, center))

    for i in range(1, 6):
        patches.append((name + str(i), center))



    for name, center in patches:
        tags = find_tags_by_name(name, all_tags)
        for tag in tags:
            counter = SequenceCounter(tag, center)

            bs_counts, bs_avg_counts = passing_sequences(center, R, cf.baseline_tag, cf)
            counts, avg_counts = passing_sequences(center, R, tag, cf)

            counter.baseline = bs_counts
            counter.baseline_avg = bs_avg_counts
            counter.patch = counts
            counter.patch_avg = avg_counts

            PIC.save_sequence(counter, counter.tag, sub_directory=cf.sub_dir)

    return


def find_tags_by_name(name:str, tags:list):
    tags = np.asarray(tags)
    idx = [name + "_" in tag for tag in tags]
    return tags[idx]


def passing_sequences(center, radius, tag:str, config):
    patches = [DOP.circular_patch(config.rows, c, radius) for c in center]
    neurons = [UNI.patch2idx(p) for p in patches]

    rate = PIC.load_rate(tag, exc_only=True, skip_warmup=True, sub_directory=config.sub_dir, config=config)
    counts = [number_of_sequences(n, avg=False, rate=rate) for n in neurons]
    avg_counts = [number_of_sequences(n, avg=True, rate=rate) for n in neurons]
    return counts, avg_counts


def number_of_sequences(neuron:(int, iter), avg:bool=False, rate:np.ndarray=None, threshold:float=None, min_dist:int=None, normalize:bool=False)->int:
    """
    Detect the number of sequences for the given neuron(s) for a given rate. Depending on the paramter threshold and min_dist.

    Parameters
    ----------
    neuron : (int, iterable)
        The neuron(s) to be analyzed.
    avg : bool, optional
        Average across neurons e.g. for a patch of neurons. The default is False.
    postfix : str, optional
        Use different rate-file. The default is None.
    threshold : float, optional
        Passed to the peak detection. The default is defined in the method.
    min_dist : int, optional
        Passed to the peak detection. The default is CF.TAU.

    Returns
    -------
    int
        DESCRIPTION.

    """
    threshold = threshold or 0.2
    min_dist = min_dist or 12

    if isinstance(neuron, int):
        number =  number_of_peaks(rate[neuron], thres=threshold, min_dist=min_dist)
    else:
        if avg:
            number =  number_of_peaks(rate[neuron].mean(axis=0), thres=threshold, min_dist=min_dist)
        else:
            number = np.zeros(len(neuron))
            for idx, n in enumerate(neuron):
                number[idx] = number_of_peaks(rate[n], thres=threshold, min_dist=min_dist)

    if normalize:
        number = number / rate.shape[1]
    return number



def number_of_peaks(data, **kwargs):
    no = putils.indexes(data, **kwargs, thres_abs=True).size
    return no


if __name__ == "__main__":
    main()
