#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author: Andreas Buttenschoen
#
from __future__ import print_function, division, absolute_import

import os

# set MKL THREADS
os.environ['MKL_VERBOSE'] = '0'
os.environ['MKL_NUM_THREADS'] = '14'

from utils.threads_mkl import set_num_threads

import numpy as np

from mol.MOL import MOL
from tdr.TDR import TDR

from tdr.Domain import Square
from model.Time import Time

from python.utils import norm1, norm2


""" For the 1d and 2d taxis tests """
def concentrationField(t, r):
    return 1. - np.cos(4. * np.pi * r)


""" Initial condition for taxis tests """
def n0(r, kappa = 0.1):
    nInit = np.zeros_like(r)
    mask1 = np.where(np.abs(r) <= 0.4 - kappa)
    mask2 = np.where((0.4 - kappa <= np.abs(r))&(np.abs(r) <= 0.4 + kappa))
    mask3 = np.where(np.abs(r) > 0.4 + kappa)
    nInit[mask1] = 1.
    nInit[mask2] = 0.5 * (1. + np.cos(np.pi * (np.abs(r[mask2]) - 0.4 + kappa)/(2. * kappa)))
    nInit[mask3] = 0.
    return nInit


class testCase(object):

    def __init__(self):
        # solver tolerance
        self.vtol = 1.e-6

        # reset solver
        self.solver = None

        # threads to run tests with
        self.threads = 14

        # 1D
        self.nop = 1

        # norm functionals
        self.norm1 = lambda x : norm1(x, self.domain.h**2)
        self.norm2 = lambda x : norm2(x, self.domain.h**2)

        # Make sure these values are reset
        self.name = 'Unknown'
        self.expected_errors = {}


    def _taxis_setup(self):
        self.L = 0.5
        self.time_round = 4
        self.domain = Square(-self.L, self.L, -self.L, self.L, n = 10)
        self.dt       = 0.0001

        # number of equations
        #
        # Equation 1: is C(x, t)
        # Equation 2: is N(x, t)
        #
        self.fieldNames = ['C(x, t)', 'N(x, t)']
        self.size = 2

        # transition matrices
        self.trans       = np.zeros((2,2))

        # set one taxis coefficient to one
        self.trans[1, 0] = 1.
        self.trans[1, 1] = 0.


    def get_ic_taxis(self):
        # create a coordinate grid
        xs = self.domain.xs()
        ys = self.domain.ys()
        xx, yy = np.meshgrid(xs, ys, sparse=True)

        _c0 = concentrationField(0, np.sqrt(xx**2 + yy**2))
        _n0 = n0(np.sqrt(xx**2 + yy**2))

        # reshape
        _c0 = np.expand_dims(_c0, axis=0)
        _n0 = np.expand_dims(_n0, axis=0)

        # assemble the initial condition
        return np.row_stack((_c0, _n0))


    def exec_solver(self, time, ic_gen):
        # Create solver object
        y0 = ic_gen()

        # since we might be using mkl use threads
        set_num_threads(self.threads)

        self.solver = MOL(TDR, y0, nop = self.nop, domain = self.domain,
                          livePlotting=False, transitionMatrix = self.trans,
                          time = time, vtol=self.vtol)

        self.solver.run()


    def test_taxis_short_time(self):
        print('setup taxis test.')
        self._taxis_setup()

        # time
        time = Time(t0 = 0., tf = 0.007, dt = 0.0001)

        # execute the solver
        print('execute the solver')
        self.exec_solver(time, self.get_ic_taxis)



if __name__ == '__main__':
    test = testCase()
    test.test_taxis_short_time()
