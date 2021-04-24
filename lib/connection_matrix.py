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



def EI_networks(landscape, nrowE, nrowI, p, stdE, stdI, shift=0, seed=0, **kwargs):
    np.random.seed()
    seed = np.random.randint(1000)
    seed = 912
    print("Seed: ", seed)
    np.random.seed(seed)
    npopE = nrowE ** 2
    npopI = nrowI ** 2

    if landscape['mode'] != 'symmetric':
        # move = cl.move(nrowE)
        from lib.move import move

        ll = cl.__dict__[landscape['mode']](nrowE, landscape.get('specs', {}))
    else:
        ll = np.zeros(npopE)

    conmatEE, conmatEI, conmatIE, conmatII = [], [], [], []
    for idx in range(npopE):
        # E -> E
        asymetric = landscape['mode'] != 'symmetric'
        # source = idx, nrowE, nrowE, int(p * npopE), stdE, False
        source = idx, nrowE, nrowE, int(p * npopE), stdE, asymetric
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        if asymetric:
            # targets = (targets + shift * move[ll[idx] % len(move)]) % npopE
            targets = move(targets, ll[idx], nrowE)
            # print(np.where(targets == idx)[0].size)
        targets = targets[targets != idx]
        hist_targets = np.histogram(targets, bins=range(npopE + 1))[0]
        conmatEE.append(hist_targets)

        # E -> I
        source = idx, nrowE, nrowI, int(p * npopI), stdE, True
        # source = idx, nrowE, ncolE, nrowI, ncolI, int(p * npopI), stdI, False
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        hist_targets = np.histogram(targets, bins=range(npopI + 1))[0]
        conmatEI.append(hist_targets)

    for idx in range(npopI):
        # I -> E
        source = idx, nrowI, nrowE, int(p * npopE), stdI, True
        # source = idx, nrowI, ncolI, nrowE, ncolE, int(p * npopE), stdE, False
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        hist_targets = np.histogram(targets, bins=range(npopE + 1))[0]
        conmatIE.append(hist_targets)

        # I -> I
        source = idx, nrowI, nrowI, int(p * npopI), stdI, False
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        targets = targets[targets != idx]
        hist_targets = np.histogram(targets, bins=range(npopI + 1))[0]
        conmatII.append(hist_targets)

    return np.array(conmatEE), np.array(conmatEI), np.array(conmatIE), np.array(conmatII), ll
