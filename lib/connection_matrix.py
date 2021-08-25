# -*- coding: utf-8 -*-
#
# connection_matrix.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib.pyplot as plt

import lib.lcrn_network as lcrn
import lib.connectivity_landscape as cl



def EI_networks(landscape, nrowE, nrowI, p, stdE, stdI, shift=0, seed=None, **kwargs):
    SYMM = 'symmetric'
    INDEPENDENT = "independent"
    if seed is None:
        np.random.seed()
        seed = np.random.randint(1000)
    else:
        np.random.seed(seed)
    npopE = nrowE ** 2
    npopI = nrowI ** 2

    if landscape['mode'] not in (SYMM, INDEPENDENT):
        # move = cl.move(nrowE)
        from lib.move import move

        ll = cl.__dict__[landscape['mode']](nrowE, landscape.get('specs', {}))
    else:
        ll = np.zeros(npopE)

    conmatEE, conmatEI, conmatIE, conmatII = [], [], [], []
    for idx in range(npopE):
        # E -> E
        asymetric = landscape['mode'] not in (SYMM, INDEPENDENT)
        # source = idx, nrowE, nrowE, int(p * npopE), stdE, False
        source = idx, nrowE, nrowE, int(p * npopE), stdE, asymetric
        if landscape['mode'] == INDEPENDENT:
            targets = lcrn.independent_targets(*source)
        else:
            targets, delay = lcrn.lcrn_gauss_targets(*source)
        if asymetric:
            # targets = (targets + shift * move[ll[idx] % len(move)]) % npopE
            targets = move(targets, ll[idx], nrowE)
        targets = targets[targets != idx]
        hist_targets = np.histogram(targets, bins=range(npopE + 1))[0]
        conmatEE.append(hist_targets)

        # E -> I
        source = idx, nrowE, nrowI, int(p * npopI), stdE, True
        # source = idx, nrowE, ncolE, nrowI, ncolI, int(p * npopI), stdI, False
        if landscape['mode'] == INDEPENDENT:
            targets = lcrn.independent_targets(*source)
        else:
            targets, delay = lcrn.lcrn_gauss_targets(*source)
        hist_targets = np.histogram(targets, bins=range(npopI + 1))[0]
        conmatEI.append(hist_targets)

    for idx in range(npopI):
        # I -> E
        source = idx, nrowI, nrowE, int(p * npopE), stdI, True
        # source = idx, nrowI, ncolI, nrowE, ncolE, int(p * npopE), stdE, False
        if landscape['mode'] == INDEPENDENT:
            targets = lcrn.independent_targets(*source)
        else:
            targets, delay = lcrn.lcrn_gauss_targets(*source)
        hist_targets = np.histogram(targets, bins=range(npopE + 1))[0]
        conmatIE.append(hist_targets)

        # I -> I
        source = idx, nrowI, nrowI, int(p * npopI), stdI, False
        if landscape['mode'] == INDEPENDENT:
            targets = lcrn.independent_targets(*source)
        else:
            targets, delay = lcrn.lcrn_gauss_targets(*source)
        targets = targets[targets != idx]
        hist_targets = np.histogram(targets, bins=range(npopI + 1))[0]
        conmatII.append(hist_targets)

    return np.array(conmatEE), np.array(conmatEI), np.array(conmatIE), np.array(conmatII), ll
