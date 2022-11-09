#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30

@author: Hauke Wernecke

Test requirements:
    baseconfig

"""


import unittest as UT

from ..baseconfig import BaseConfig

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from libimport pickler as PIC



tag_0_0 = 'proxy-0_1_10_15_0'
tag_0_1 = 'proxy-0_1_10_15_1'
TAGS_0 = [tag_0_0, tag_0_1]
tag_1_0 = 'proxy-1_1_10_15_0'
tag_1_1 = 'proxy-1_1_10_15_1'
TAGS_1 = [tag_1_0, tag_1_1]

TAGS_FLAT = [*TAGS_0, *TAGS_1]
TAGS_DEEP = [TAGS_0, TAGS_1]
TAGS_SEED_0 = [tag_0_0, tag_1_0]


class TestBaseConfig(UT.TestCase):

    def setUp(self):
        self.config = BaseConfig()
        self.config.center_range = OrderedDict({
            "proxy-0": (0, 0),
            "proxy-1": (1, 1),
        })
        self.config.RADIUSES = 1,
        self.config.AMOUNT_NEURONS = 10,
        self.config.PERCENTAGES = .15,


    def tearDown(self):
        pass


    def test_get_all_tags(self):
        # Checks against all seeds (0, 1)
        self.assertCountEqual(self.config.get_all_tags(), TAGS_FLAT)


        proxy_seeds = None
        tags = self.config.get_all_tags(seeds=proxy_seeds)
        self.assertCountEqual(tags, TAGS_FLAT)

        proxy_seeds = 0
        tags = self.config.get_all_tags(seeds=proxy_seeds)
        self.assertEqual(tags, TAGS_SEED_0)

        proxy_seeds = "all"
        tags = self.config.get_all_tags(seeds=proxy_seeds)
        self.assertCountEqual(tags, TAGS_DEEP)
        print(tags)



    def test__seeds_and_method(self):
        proxy_list = []

        proxy_seeds = None
        seeds, method = self.config._seeds_and_method(proxy_seeds, proxy_list)
        self.assertCountEqual(self.config.simulation_seeds, seeds)
        self.assertEqual(method, proxy_list.extend)
        proxy_seeds = 0
        seeds, method = self.config._seeds_and_method(proxy_seeds, proxy_list)
        self.assertCountEqual([0], seeds)
        self.assertEqual(method, proxy_list.extend)
        proxy_seeds = "all"
        seeds, method = self.config._seeds_and_method(proxy_seeds, proxy_list)
        self.assertCountEqual(self.config.simulation_seeds, seeds)
        self.assertEqual(method, proxy_list.append)

        # TODO: Raise error it seeds are not  in config.
        proxy_seeds = 123



if __name__ == '__main__':
    UT.main()
