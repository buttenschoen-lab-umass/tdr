#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import numpy as np
from tdr.helpers import asarray, round_to_nearest_fraction
from tdr.Boundary import DomainBoundary, StichTogether

from SimulationObject.SimulationObject import SimulationObject
from SimulationObject.SimulationObjectFactory import createSimObject
from utils.xml import isParameter, isDomainBoundary


"""
    An object representing a 1D interval. In mathematical terms this represents
    a closed interval [x0, xf]
"""
class Interval(SimulationObject):

    __short_name__ = 'int'

    def __init__(self, a = 0, b = 1, *args, **kwargs):
        super(Interval, self).__init__(*args, **kwargs)

        # set the basic parameters
        self.name               = kwargs.pop('name', 'x')

        # save actual boundaries
        self.a = a
        self.b = b

        # in general stored like
        # [ start_patch1, end_patch1]
        # [ start_patch2, end_patch2]
        self.endPoints          = np.expand_dims(np.array([a, b]), axis=0)

        # number of patches -> in 1D always one
        self.nop                = int(kwargs.pop('nop', 1))

        # check if we have a xml node
        xml = kwargs.pop('xml', None)

        # Length parameters
        self.n                  = np.int(kwargs.pop('n', 6))
        self.cellsPerUnitLength = np.int(kwargs.pop('cellsPerUnitLength', 2**self.n))
        self.M                  = np.int(self.cellsPerUnitLength)
        self.N                  = kwargs.pop('N', None)
        self.h                  = 1. / self.cellsPerUnitLength

        # use fixed N
        self.useFixedN          = self.N is not None

        # for plotting
        self.y0                 = kwargs.pop('y0', 0.)
        self.yf                 = kwargs.pop('yf', 10.)

        self.bd                 = kwargs.pop('bd', DomainBoundary(dim=self.dimensions))

        if xml is not None:
            print('Creating Interval from xml!')
            self._create_from_xml(xml, *args, **kwargs)

        # call reset
        self._reset()


    """ Properties """
    @property
    def x0(self):
        return np.min(self.endPoints[:, 0])


    @property
    def x0s(self):
        return self.endPoints[:, 0]


    @property
    def xf(self):
        return np.max(self.endPoints[:, 1])


    @property
    def xfs(self):
        return self.endPoints[:, 1]


    """ Creation """
    def _create_from_args(self, *args, **kwargs):
        pass


    def _create_from_xml(self, xml, *args, **kwargs):
        # set name
        setattr(self, 'name', xml.tag)

        # first check if the main node has attributes
        for name, value in xml.attrib.items():
            setattr(self, name, value)

        parameters = []
        for child in xml:
            p = createSimObject(child)
            if isParameter(child):
                parameters.append(p)
            elif isDomainBoundary(child):
                self.bd = p
            else:
                assert False, 'Encountered unknown xml type %s' % child.tag

        # set the objects parameters!
        for p in parameters:
            # HACK FIXME
            if p.name == 'x0':
                self.endPoints[0] = p.value
            elif p.name == 'xf':
                self.endPoints[1] = p.value
            else:
                setattr(self, p.name, p.value)


    """ Factory """
    class Factory:
        def create(self, *args, **kwargs):
            return Interval(*args, **kwargs)


    def set_y(self, y):
        self.endPoints = y


    """ Internal methods """
    def _setup_bds(self):
        self.bds = self.nop * [None]

        # if we only have one patch -> don't do anything special
        if self.nop == 1:
            self.bds[0] = self.bd
            return

        # if we have more than one patch -> create smaller subdomains
        for i in range(self.nop):
            if i == 0:
                bd = DomainBoundary(dim=self.dimensions, left=self.bd.left, right=StichTogether(1))
            elif i == self.nop-1:
                bd = DomainBoundary(dim=self.dimensions, left=StichTogether(-1), right=self.bd.right)
            else:
                bd = DomainBoundary(dim=self.dimensions, left=StichTogether(-1.), right=StichTogether(1.))
            self.bds[i] = bd


    def _reset(self):
        # at the moment assume uniform partitions when nop is larger one
        self.L      = np.abs(self.xf - self.x0)
        self.L_part = self.L / self.nop

        if self.useFixedN:
            assert self.nop==1, 'not tested with more than one patch!'
            self.cellsPerUnitLength = self.N / self.L
            assert self.N == np.int(self.L * self.cellsPerUnitLength), ''
        else:
            # in the case that we have patches -> we must have that each patch
            # has an equal number of sub-divisions. To ensure this we simply
            # increases N per patch until it's divisible by the number of
            # patches.
            requested_N_patch = np.int(self.L * self.cellsPerUnitLength)

            while requested_N_patch % self.nop != 0:
                requested_N_patch += 1
            self.N = np.tile(np.int(requested_N_patch / self.nop), self.nop)

            # must update cellsPerUnitLength!
            self.cellsPerUnitLength = requested_N_patch / self.L


        self.h                  = 1. / self.cellsPerUnitLength
        self.dX                 = np.tile(self.h, self.nop)

        # must update endpoints
        self.endPoints = np.zeros((self.nop, 2))
        patchBd        = np.linspace(self.a, self.b, 1+self.nop, endpoint=True)
        for i in range(self.nop):
            self.endPoints[i, :] = np.asarray([patchBd[i], patchBd[i+1]])

        # setup bds
        self._setup_bds()

        # now compute and cache x
        self.x                  = self.xs()

        assert np.all(self.dX > 0), 'Interval must have non-negative step size.'
        assert np.all(self.N > 1), 'Interval N = %s must be larger than 1.' % self.N
        assert self.L > 0, 'Interval must have non-negative length.'


    """ Public methods """
    def origin(self):
        return np.asarray(self.x0)


    @property
    def dimensions(self):
        return 1


    def xs(self):
        return np.linspace(self.x0, self.xf, np.sum(self.N), endpoint=True)


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
        return 'Interval(%.2f, %.2f, %.4g, %s, %s, %s)' \
                % (self.x0, self.xf, self.h, self.N, self.bd.left.name, self.bd.right.name)


    def __str__(self):
        return self.__repr__()


    """ Mainly for the parsing functions """
    def __call__(self):
        return self.x


    def getIndex(self, value):
        new_value = int(value / self.dX[0])
        return int(new_value) % int(self.N)


    """ deformation support """
    def deform(self, newPosition):
        self.endPoints  = newPosition
        self.x          = self.xs()
        self.h          = self.x[1] - self.x[0]
        self.dX         = asarray(self.h)


    """ Update all the interval information """
    def update(self):
        self.x          = self.xs()
        self.h          = self.x[1] - self.x[0]
        self.dX         = asarray(self.h)



"""
    An object representing a 2D square. In mathematical terms this represents
    a closed interval [x0, xf] x [y0, yf]
"""
class Square(object):
    def __init__(self, a, b, c, d, *args, **kwargs):
        # set the basic parameters
        self.x0                 = a
        self.xf                 = b
        self.y0                 = c
        self.yf                 = d

        # number of patches -> in 1D always one
        self.nop                = 1

        # Length parameters
        self.n                  = kwargs.pop('n', 8)
        self.cellsPerUnitLength = kwargs.pop('cellsPerUnitLength', 2**self.n)
        self.h                  = 1. / self.cellsPerUnitLength

        self.bd                 = kwargs.pop('bd', DomainBoundary(dim=self.dimensions))

        # call reset
        self._reset()


    """ Internal methods """
    def _reset(self):
        self.L          = np.array([self.xf - self.x0, self.yf - self.y0])
        self.N          = np.asarray([self.xf - self.x0, self.yf - self.y0]) * self.cellsPerUnitLength
        self.N          = self.N.astype(np.int)
        self.dX         = self.h * np.ones(2)


    """ Public methods """
    def origin(self):
        return np.asarray([self.x0, self.y0])


    @property
    def dimensions(self):
        return 2


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


