#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division, absolute_import

import numpy as np
from tdr.Boundary import DomainBoundary


"""
    An object representing a 2D square. In mathematical terms this represents
    a closed interval [x0, xf] x [y0, yf]
"""
class Square(object):
    def __init__(self, a, b, c, d, *args, **kwargs):
        # in general stored like
        # [ start_patch1, end_patch1]
        # [ start_patch2, end_patch2]
        self.endPoints          = np.expand_dims(np.array([[a, b], [c, d]]), axis=0)

        # number of patches -> in 1D always one
        self.nop                = 1

        # Length parameters
        self.n                  = kwargs.pop('n', 8)
        self.cellsPerUnitLength = kwargs.pop('cellsPerUnitLength', 2**self.n)
        self.M                  = np.int(self.cellsPerUnitLength)
        self.h                  = 1. / self.cellsPerUnitLength

        self.bd                 = kwargs.pop('bd', DomainBoundary(dim=self.dimensions))

        # call reset
        self._reset()


    """ Internal methods """
    def _reset(self):
        self.L          = np.array([self.xf - self.x0, self.yf - self.y0])
        self.N          = np.asarray([self.xf - self.x0, self.yf - self.y0]) * self.cellsPerUnitLength
        self.N          = np.tile(self.N.astype(np.int), self.nop)
        self.dX         = self.h * np.ones(2)
        self.bds        = self.nop * [None]

        # if we only have one patch -> don't do anything special
        if self.nop == 1:
            self.bds[0] = self.bd

        self.x = self.xs()
        self.y = self.ys()


    """ Properties """
    @property
    def origin(self):
        return np.asarray([self.x0, self.y0])

    @property
    def dimensions(self):
        return 2

    @property
    def x0(self):
        return np.min(self.endPoints[:, 0, :])

    @property
    def x0s(self):
        return self.endPoints[:, 0, :]

    @property
    def y0(self):
        return np.min(self.endPoints[:, 1, :])

    @property
    def y0s(self):
        return self.endPoints[:, 1, :]

    @property
    def xf(self):
        return np.max(self.endPoints[:, 0, :])

    @property
    def xfs(self):
        return self.endPoints[:, 0, :]

    @property
    def yf(self):
        return np.max(self.endPoints[:, 1, :])

    @property
    def yfs(self):
        return self.endPoints[:, 1, :]

    def xs(self):
        return np.linspace(self.x0, self.xf, self.N[0], endpoint=True)

    def ys(self):
        return np.linspace(self.y0, self.yf, self.N[1], endpoint=True)

    def box(self):
        return [self.x0, self.xf, self.y0, self.yf]

    def resize(self, arr):
        assert False, 'Not implemented for the 2D-square'


    def size(self):
        return int(np.sum(self.N))


    def __repr__(self):
        return 'Square(%.2f, %.2f, %.2f, %.2f, %s, %s, %s, %s, %s)' % \
                (self.x0, self.xf, self.y0, self.yf, self.N, self.bd.left.name,
                self.bd.right.name, self.bd.bottom.name, self.bd.top.name)


    def __str__(self):
        return self.__repr__()


    """ deformation support """
    def deform(self, t):
        assert False, 'not implemented'


    def setup_deformation(self, functional):
        assert False, 'not implemented'


