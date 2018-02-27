#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function

import matplotlib
matplotlib.use('GTKCairo')
print("Using backend %s." % matplotlib.get_backend())

import numpy as np
#from utils import Average
from mol.MOL import MOL
from tdr.TDR import TDR
from tdr.Domain import Interval
from tdr.helpers import getHillFunction, getStepFunction
from mol.Time import Time

from python.utils import Average


if __name__ == '__main__':
    n = 8

    # 1 length unit = 1[um], time = 1[s]
    interval = Interval(0, 10, n = n)
    x = interval.xs()

    # number of equations
    no_eqs = 2

    # total mass in the system
    K = 2.8

    # Generate initial condition
    y0 = np.empty([no_eqs, x.size])
    y0[0, :] = getStepFunction(x, step_point = 1.)
    y0[1, :] = 1. + np.random.normal(loc = 5., scale=.25, size=x.size)

    mass = Average(x, y0[0, :]) + Average(x, y0[1, :])
    y0[0, :] *= (K / mass)
    y0[1, :] *= (K / mass)
    y0 = y0.reshape(no_eqs * x.size)

    # transition matrices
    # 1) diffusion
    trans = np.array([[0.1, 0.], [0., 10.]])

    # 2) Reaction terms
    I = 1.
    inActivation = lambda G, Gi : I * G

    # model constants
    b     = 0.
    gamma = 1.

    nHill = 2
    hill = getHillFunction(nHill)
    Activation   = lambda G, Gi : Gi * (b + gamma * hill(G))

    nonLinG    = lambda G, Gi : Activation(G, Gi) - inActivation(G, Gi)
    nonLinGi   = lambda G, Gi : - nonLinG(G, Gi)

    reaction = np.array([nonLinG, nonLinGi])

    # times
    t = Time(tf = 10., dt = 0.25)

    # solver for internal dynamics
    solver = MOL(TDR, y0, domain = interval, livePlotting = True,
                 noPDEs = no_eqs, transitionMatrix = trans,
                 reactionTerms = reaction, time = t, vtol = 1.e-3)

    solver.run()

    # NOW WE GROW THE DOMAIN
    solver.resize()









