#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 2021

@author: Hauke Wernecke
"""
import numpy as np
import matplotlib.pyplot as plt
import cflogger
log = cflogger.getLogger()

import util.pickler as PIC
import dopamine as DOP
import universal as UNI

import peakutils as putils
from plot.lib import SequenceCounter

from analysis import SequenceDetector

### Idea: Plot the number of sequences passed through a patch

# TODO: Add a baseline level as first data point

# TODO: Add some discrimination line?
# TODO: Reduce the number of data points to a specific parameter of interest.
# TODO: Uniform scale



R = 2
THRESHOLD = .2
MINIMAL_PEAK_DISTANCE = 12

def main():
    from params import PerlinConfig
    cf = PerlinConfig()

    all_tags = cf.get_all_tags()
    log.info(f"Analysis of passing sequences: Tags are: {all_tags}")
    log.info(f"Analysis of passing sequences: Center names are: {cf.center_range.keys()}")

    patches = []

    center = ((30, 18), (28, 26), )
    # UNI.append_spot(patches, "in", (center)
    # UNI.append_spot(patches, "edge", (center)
    # UNI.append_spot(patches, "out", (center)
    
    # UNI.append_spot(patches, "linker", ((21, 65), (30, 61), ))
    # UNI.append_spot(patches, "repeater", ((2, 31), (29, 35), (29, 25)))

    center = ((35, 49), (49, 36), )
    # UNI.append_spot(patches, "in-activator", (center)
    # UNI.append_spot(patches, "edge-activator", (center)
    UNI.append_spot(patches, "out-activator", (center))

    UNI.append_spot(patches, "starter", ((47, 4), (48, 8)))

    for name, center in patches:
        tags = cf.get_all_tags([name])
        for tag in tags:
            counter = SequenceCounter(tag, center)

            counter.baseline, counter.baseline_avg = passing_sequences(center, R, cf.baseline_tag, cf)
            counter.patch, counter.patch_avg = passing_sequences(center, R, tag, cf)

            PIC.save_sequence(counter, counter.tag, sub_directory=cf.sub_dir)

    return

def passing_sequences(center, radius, tag:str, config):
    rate = PIC.load_rate(tag, exc_only=True, skip_warmup=True, sub_directory=config.sub_dir, config=config)
    
    sd = SequenceDetector(R, THRESHOLD, MINIMAL_PEAK_DISTANCE)
    return sd.passing_sequences(rate, center, rows=config.rows)


if __name__ == "__main__":
    main()
    plt.show()