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


from  dataclasses import dataclass
import os
import pickle
import numpy as np
from pathlib import Path

from params.baseconfig import FIGURE_DIR, FIGURE_SUFFIX, FIGURE_ALTERNATIVE_SUFFIX, ANIMATION_SUFFIX
from params.baseconfig import DATA_DIR, AVG_TAG
import lib.universal as UNI

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""



#===============================================================================
# CLASS
#===============================================================================

@dataclass
class Pickler:
    config: object

    def get_fig_filename(self, tag:str, format_ = FIGURE_ALTERNATIVE_SUFFIX) -> str:
        fname = self.prepend_dir(tag, directory=FIGURE_DIR)
        return fname + format_


    def save_animation(self, filename:str, animation:object):
        """
        Saves the animation in the subdirectory of the config.
        """
        filename = self.prepend_dir(filename, self.config.sub_dir)
        filename = self.prepend_dir(filename, FIGURE_DIR)
        self.create_dir(filename)
        animation.save(filename + ANIMATION_SUFFIX)


    def save_figure(self, filename:str, figure:object):
        """
        Saves the figure in the subdirectory of the config.
        """
        filename = self.prepend_dir(filename, self.config.sub_dir)
        filename = self.prepend_dir(filename, FIGURE_DIR)
        self.create_dir(filename)

        figure.savefig(filename + FIGURE_SUFFIX)
        figure.savefig(filename + FIGURE_ALTERNATIVE_SUFFIX)


    @staticmethod
    def prepend_dir(filename: str, directory: str = DATA_DIR):
        return os.path.join(directory, filename)


    @staticmethod
    def create_dir(filename:str):
        """Creates directories such that the filename is valid."""
        path = Path(filename)
        os.makedirs(path.parent.absolute(), exist_ok=True)



if __name__ == '__main__':
    main()
