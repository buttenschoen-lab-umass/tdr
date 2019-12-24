#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import absolute_import, print_function, division

import os

import numpy as np
from collections import OrderedDict

from mol.MOL import MOL
from tdr.TDR import TDR
from tdr.Boundary import DomainBoundary, Periodic
from tdr.Domain import Interval
from model.Time import Time
from python.utils import Average


if __name__ == '__main__':

    # Number of spatial subdivisions per unit length as a power of 2
    n = 8

    # domain length
    L = 10

    bd = DomainBoundary(left=Periodic(), right=Periodic())
    interval = Interval(0, L, n=n, bd=bd)

    # create some initial condition
    x  = interval.xs()
    y0 = 1. + np.random.normal(loc=0, scale=0.05, size=x.size)

    # Diffusion - Taxis matrix
    trans    = np.ones(1).reshape((1,1))

    # Non-local adhesion matrix
    alpha = 5
    Adhtrans = alpha * np.ones(1).reshape((1,1))

    # times
    t0 = 0
    tf = 10

    # number of data points to save
    no_data_points = 10

    print(tf, t0, no_data_points)
    dt = max((tf - t0) / no_data_points, 0.01)
    time = Time(t0 = t0, tf = tf, dt = dt)

    # setup output paths for data?
    name = 'adhesion_example' + '_L=%f_alpha=%f_tf=%f' % (L, alpha, tf)
    print('Saving to: %s.' % name)

    solver = MOL(TDR, y0, domain=interval,
                 transitionMatrix=trans,
                 AdhesionTransitionMatrix=Adhtrans,
                 time=time, name=name, save=False,
                 verbose=True)

    solver.run()
