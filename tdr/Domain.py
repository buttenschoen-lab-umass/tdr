#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

import numpy as np
from utils import asarray
from Boundary import DomainBoundary

def round_to_nearest_fraction(number, n = 4):
    fraction = 2**n
    val = number * fraction
    val = np.ceil(val)
    return val / fraction


"""
    1-D only for the moment.
"""
class Interval(object):
    def __init__(self, a, b, *args, **kwargs):
        # set the basic parameters
        self.x0                 = a
        self.xf                 = b

        # Length parameters
        self.n                  = kwargs.pop('n', 5)
        self.cellsPerUnitLength = kwargs.pop('cellsPerUnitLength', 2**self.n)

        # for plotting
        self.y0                 = kwargs.pop('y0', 0.)
        self.yf                 = kwargs.pop('yf', 10.)

        self.bd                 = kwargs.pop('bd', DomainBoundary())

        # call reset
        self._reset()


    """ Internal methods """
    def _reset(self):
        self.L                  = np.abs(self.xf - self.x0)
        self.h                  = 1. / self.cellsPerUnitLength
        self.N                  = self.L * self.cellsPerUnitLength
        self.dX                 = asarray(self.h)


    """ Public methods """
    def xs(self):
        return np.linspace(self.x0, self.xf - self.h, self.N)


    def box(self):
        return [self.x0, self.xf, self.y0, self.yf]


    def resize(self, arr):
        self.x0 = round_to_nearest_fraction(arr[0], self.n)
        self.xf = round_to_nearest_fraction(arr[1], self.n)
        N_old = self.N
        self._reset()
        return N_old


    def size(self):
        return int(self.N)


    def __repr__(self):
        return 'Interval(%.2f, %.2f, %d)' % (self.x0, self.xf, self.N)


    def __str__(self):
        return self.__repr__()


