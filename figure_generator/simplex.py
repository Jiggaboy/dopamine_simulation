#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

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
from matplotlib import rcParams
from params import config

import lib.pickler as PIC
from lib.connectivitymatrix import ConnectivityMatrix
import lib.universal as UNI
from figure_generator.in_out_degree import plot_shift_arrows


rcParams["font.size"] = 12
rcParams["figure.figsize"] = (3.5, 3.5)


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    force = UNI.yes_no("Force new connectivity matrix?", False)

    save = True

    config.rows = 26
    config.landscape.params["size"] = 1
    config.landscape.params["base"] = 8

    force = UNI.yes_no("Force new connectivity matrix?", False)
    conn = ConnectivityMatrix(config, force=force)

    fig, ax = plt.subplots(num="simplex_noise")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plot_shift_arrows(conn.shift)
    bins = 6
    plt.xlim(-0.5, bins-0.5)
    plt.xticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
    plt.ylim(-0.5, bins-0.5)
    plt.yticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
    # plt.tight_layout()
    if save:
        PIC.save_figure(fig.get_label(), fig, transparent=True)

#===============================================================================
# METHODS
#===============================================================================





if __name__ == '__main__':
    main()
    plt.show()
