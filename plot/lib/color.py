#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


def _get_colors(number:int, cmamp:str="gist_rainbow"):
    cm = plt.get_cmap(cmamp)
    return [cm(1. * i / number) for i in range(number)]
