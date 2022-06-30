#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prints the time elapsed for a function call.

Created on 2022-06-29

@author: Hauke Wernecke
"""

from functools import wraps, partial
from time import perf_counter

def functimer(func=None, *, logger=None):
    ''' Measures the elapsed time and prints it (optionally the same as a logger output.'''
    if func is None:
        return partial(functimer, logger=logger)
    
    
    printer = print if logger is None else logger.error
    
    @wraps(func)
    def wrapper(*args, **kwargs):                                                                             
        pre = perf_counter()  
        func(*args, **kwargs)
        post = perf_counter()
        printer(f"Time elapsed ({func.__name__}): {post - pre}")
    return wrapper



if __name__ == "__main__":
    import logging
    logger = logging.getLogger()

    @functimer(logger=logger)
    def with_logger():                                                                                                 
        print("Logger... Check")

    @functimer()
    def without_logger():                                                                                        
        print("Without Logger... Check")

    @functimer
    def without_parentheses():                                                                              
        print("Without Parentheses... Check")
        
    with_logger()
    without_logger()
    without_parentheses()
                                        

