#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:55:40 2021

@author: hauke
"""

import numpy as np

def move(idx, direction:int, nrow:int, *args, **kwargs):
    shift_y = 0
    shift_x = 0
    if direction == 0:
        shift_y = 0
        shift_x = 1
    elif direction == 1:
        shift_y = 1
        shift_x = 1
    elif direction == 2:
        shift_y = 1
        shift_x = 0
    elif direction == 3:
        shift_y = 1
        shift_x = -1
    elif direction == 4:
        shift_y = 0
        shift_x = -1
    elif direction == 5:
        shift_y = -1
        shift_x = -1
    elif direction == 6:
        shift_y = -1
        shift_x = 0
    elif direction == 7:
        shift_y = -1
        shift_x = 1
    trow = np.floor(idx / nrow)
    target = nrow * (trow + shift_y) % (nrow*nrow) + (idx + shift_x) % nrow
    if isinstance(idx, int):
        target = int(target)
    else:
        target = target.astype(int, copy=False)
    return target


def test(source, target, direction):
    assert move(source, direction=direction, nrow=70) == target, f"Expected: {target}"


targets = np.array([0, 69, 4899])
assert all(move(targets, 0, 70) == np.array([1, 0, 4830]))

# nrows fixed to 70
direction = 0
test(source=0, target=1, direction=direction)
test(source=20, target=21, direction=direction)
test(source=90, target=91, direction=direction)
test(source=69, target=0, direction=direction)
test(source=139, target=70, direction=direction)
test(source=4899, target=4830, direction=direction)

direction = 1
test(source=0, target=71, direction=direction)
test(source=20, target=91, direction=direction)
test(source=90, target=161, direction=direction)
test(source=69, target=70, direction=direction)
test(source=139, target=140, direction=direction)
test(source=4899, target=0, direction=direction)

direction = 2
test(source=0, target=70, direction=direction)
test(source=20, target=90, direction=direction)
test(source=90, target=160, direction=direction)
test(source=69, target=139, direction=direction)
test(source=139, target=209, direction=direction)
test(source=4899, target=69, direction=direction)

direction = 3
test(source=0, target=139, direction=direction)
test(source=20, target=89, direction=direction)
test(source=90, target=159, direction=direction)
test(source=69, target=138, direction=direction)
test(source=139, target=208, direction=direction)
test(source=4899, target=68, direction=direction)

direction = 4
test(source=0, target=69, direction=direction)
test(source=20, target=19, direction=direction)
test(source=90, target=89, direction=direction)
test(source=69, target=68, direction=direction)
test(source=139, target=138, direction=direction)
test(source=4899, target=4898, direction=direction)

direction = 5
test(source=0, target=4899, direction=direction)
test(source=20, target=4849, direction=direction)
test(source=90, target=19, direction=direction)
test(source=69, target=4898, direction=direction)
test(source=139, target=68, direction=direction)
test(source=4899, target=4828, direction=direction)

direction = 6
test(source=0, target=4830, direction=direction)
test(source=20, target=4850, direction=direction)
test(source=90, target=20, direction=direction)
test(source=69, target=4899, direction=direction)
test(source=139, target=69, direction=direction)
test(source=4899, target=4829, direction=direction)

direction = 7
test(source=0, target=4831, direction=direction)
test(source=20, target=4851, direction=direction)
test(source=90, target=21, direction=direction)
test(source=69, target=4830, direction=direction)
test(source=139, target=0, direction=direction)
test(source=4899, target=4760, direction=direction)
