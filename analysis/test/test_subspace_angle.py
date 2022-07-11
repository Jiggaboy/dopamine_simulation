#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30

@author: Hauke Wernecke

Test requirements:
    PerlinConfig
    Data of the corresponding config with the patches starter and edge-activator

"""


import unittest as UT

from analysis import SubspaceAngle

from params import PerlinConfig
import dopamine as DP



class TestSubspaceAngle(UT.TestCase):
    cfg = PerlinConfig()
    
    tag = "starter", "edge-activator"
    MAX_components = 5
    THR_components = 15
    
    LOCAL_NEURONS = DP.circular_patch(cfg.rows, center=(29, 29), radius=6)

    
    def setUp(self):
        self.tags = self.find_tags(self.tag)
        self.angle = SubspaceAngle(self.cfg)
    
    
    def find_tags(self, t:tuple)->list:
        tags = []
        for tag_name in self.tag:
            tags.extend([t for t in self.cfg.get_all_tags() if t.startswith(tag_name)])
        return tags    
    
    
    def test_patch_vs_baseline(self):
        for i in range(1, self.MAX_components):
            self.angle.fit(self.tags[0], n_components=i)
    
    
    def test_patch_vs_patch(self):
        for i in range(1, self.MAX_components):
            self.angle.fit(*self.tags, n_components=i)
            
    
    def test_local(self):
        for i in range(1, self.MAX_components):
            self.angle.fit(self.tags[0], n_components=i, mask=self.LOCAL_NEURONS)
            
            
    def test_find_min_components(self):
        self.angle.pseudo_fit(self.tags[0], n_components=self.THR_components, mask=self.LOCAL_NEURONS)
        self.angle.find_min_components()
    
            
            
    
            
if __name__ == '__main__':
    UT.main()