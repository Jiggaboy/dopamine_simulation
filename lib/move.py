#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:55:40 2021

@author: hauke
"""

import numpy as np


def move(idx, direction: int, nrow: int, *args, **kwargs):
    trow = np.floor(idx / nrow)
    target = nrow * (trow + _shift_y(direction)) % (nrow * nrow) + (idx + _shift_x(direction)) % nrow
    if isinstance(idx, int):
        target = int(target)
    else:
        target = target.astype(int, copy=False)
    return target


def _shift_x(direction:int):
    if direction in (0, 1, 7):
        return 1
    elif direction in (3, 4, 5):
        return -1
    else:
        return 0


def _shift_y(direction:int):
    if direction in (1, 2, 3):
        return 1
    elif direction in (5, 6, 7):
        return -1
    else:
        return 0


def get_shift(direction:int, possible_directions:int = 8):
    if direction is None:
        return np.zeros(2)
    phase = 2 * np.pi / possible_directions * direction
    shift = np.exp(1j * phase)
    return np.asarray((shift.real, shift.imag))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.figure()
    for i in np.arange(8):
        shift = get_shift(i)
        plt.scatter(*shift)




    def test(source, target, direction):
        assert move(source, direction=direction, nrow=70) == target, f"Expected: {target}"



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
