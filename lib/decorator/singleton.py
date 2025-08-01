#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Implements the SingletonClass which can be derived.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'


#===============================================================================
# CLASS
#===============================================================================

class SingletonClass(object):
  def __new__(cls, *args, **kwargs):
    if not hasattr(cls, '_instance'):
      cls._instance = super(SingletonClass, cls).__new__(cls)
    return cls._instance
