#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    - Can request animations.

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import lib.pickler as PIC




def _average_rate(*tags, **save_params):
    """Averages the rates of the given tags. Saves the averaged rates."""
    for tag in tags:
        try:
            rate = PIC.load_rate(tag, exc_only=True, **save_params)
        except FileNotFoundError:
            logger.error(f"Averaging failed! Could not find file to the tag: {tag}")
            continue
        avgRate = rate.mean(axis=1)
        PIC.save_avg_rate(avgRate, tag, **save_params)
