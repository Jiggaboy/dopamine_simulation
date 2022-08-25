#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30

@author: Hauke Wernecke

Test requirements:
    PerlinConfig
    Data of the corresponding config with the patches starter and repeater

"""


import unittest as UT

from analysis import SubspaceAngle

from params import PerlinConfig
import dopamine as DP



class TestSubspaceAngle(UT.TestCase):
    cfg = PerlinConfig()
    
    tag = "repeater", "starter"
    MAX_components = 5
    THR_components = 1000
    
    LOCAL_NEURONS = DP.circular_patch(cfg.rows, center=(29, 29), radius=24)

    @property
    def seed(self):
        return self.cfg.drive.seeds[0]
    
    
    def setUp(self):
        self.tags = self.cfg.get_all_tags(self.tag, seeds=self.seed)
        self.angle = SubspaceAngle(self.cfg)
        
    
    def test_patch_vs_baseline(self):
        for i in range(1, self.MAX_components):
            self.angle.fit(self.tags[0], self.cfg.baseline_tag(self.seed), n_components=i)
    
    
    def test_patch_vs_patch(self):
        for i in range(1, self.MAX_components):
            self.angle.fit(*self.tags[:2], n_components=i)
            
            
    def test_local(self):
        for i in range(1, self.MAX_components):
            self.angle.fit(self.tags[0], self.cfg.baseline_tag(self.seed), n_components=i, mask=self.LOCAL_NEURONS)
    
            
            
    
            
if __name__ == '__main__':
    UT.main()