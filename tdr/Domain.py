#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

import numpy as np
from utils import asarray
from Boundary import DomainBoundary

"""
    1-D only for the moment.
"""
class Interval(object):
    def __init__(self, a, b, *args, **kwargs):
        self.L                  = np.abs(b - a)
        self.n                  = kwargs.pop('n', 5)
        self.cellsPerUnitLength = kwargs.pop('cellsPerUnitLength', 2**self.n)
        self.h                  = 1. / self.cellsPerUnitLength
        self.N                  = self.L * self.cellsPerUnitLength
        self.x0                 = a
        self.xf                 = b
        self.dX                 = asarray(self.h)

        # for plotting
        self.y0                 = kwargs.pop('y0', 0.)
        self.yf                 = kwargs.pop('yf', 10.)

        self.bd                 = kwargs.pop('bd', DomainBoundary())


    def xs(self):
        return np.linspace(self.x0, self.xf - self.h, self.N)


    def box(self):
        return [self.x0, self.xf, self.y0, self.yf]

