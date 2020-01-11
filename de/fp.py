#!/usr/bin/python
# -*- coding: utf-8 -*-
# author: Andreas Buttenschoen 2019
from __future__ import print_function, division

import numpy as np

from scipy import optimize
from itertools import product


def check_in_range(fp, x_range, y_range):
    if fp[0] < x_range[0] or fp[0] > x_range[1]:
        return False

    if fp[1] < y_range[0] or fp[1] > y_range[1]:
        return False

    return True


def fp(f, pars, x_range=None, y_range=None, rdecimals=4, npts=25):
    """Return fixed points of f."""

    if x_range is None:
        x_range = [5.5, 8]

    if y_range is None:
        y_range = [0.1, 3]

    xs = np.linspace(x_range[0], x_range[1], npts)
    ys = np.linspace(y_range[0], y_range[1], npts)

    # let's see what the range of f is
    #f2 = lambda x, y, pars : f(np.array([x, y]), *pars)
    #p, q = np.meshgrid(x_range, y_range)
    #z    = f2(p, q, pars)

    fps = []
    for x, y, in product(xs, ys):
        if pars is None:
            fp, info, ier, mesg = optimize.fsolve(f, [x, y], full_output=True)
        else:
            fp, info, ier, mesg = optimize.fsolve(f, [x, y], args=pars, full_output=True)

        # only include zeros that have really converged!
        if ier == 1 and check_in_range(fp, x_range, y_range):
            fps.append(fp)

    if len(fps) > 0:
        fps = np.vstack(fps)
        fps = np.unique(fps.round(decimals=rdecimals), axis=0)

    return fps
