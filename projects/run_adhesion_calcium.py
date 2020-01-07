#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import os, sys
from datetime import datetime
import numpy as np

# append system path
sys.path.append('/home/adrs0061/sources/NumericalPDEs')
sys.path.append('/home/adrs0061/sources/NumericalPDEs/python')

from mol.MOL import MOL
from tdr.Domain import Interval
from tdr.Boundary import DomainBoundary, Periodic
from tdr.TDR import TDR
from model.Time import Time

homedir  = os.path.expanduser('~')
basepath = os.path.join(homedir,  'sources', 'NumericalPDEs')
basename ='AdhesionCalcium'
datapath = os.path.join(basepath, 'results', 'adhesion-calcium', basename)

if not os.path.exists(datapath):
    os.makedirs(datapath)

# save startime
startTime = datetime.now()

L = 120
n = 6

bd       = DomainBoundary(left=Periodic(), right=Periodic())
interval = Interval(0, L, n=n, bd=bd)

# We have three concentrations
#
#   - c(x, t): Calcium as a function of space and time
#   - h(x, t): ??
#   - u(x, t): Cell Density
#

# Diffusion coefficients
Dc = 0.1
Dh = 0.0
Du = 0.0025

# create diffusion - taxis matrix
trans = np.zeros((3, 3))

trans[0, 0] = Dc
trans[1, 1] = Dh
trans[2, 2] = Du

# non-local transition matrix
adhTrans = np.zeros((3, 3), dtype=object)

# non-local adhesion parameters
s_star = 1.0
a1     = 0.5
a2     = 0.5
Rs     = 5.0

# This function is a ToDO!
def S(y):
    c = y[0, :]
    # h = y[1, :]
    # u = y[2, :]
    #print('c:', c)
    ret = s_star * (1. - a1 * c / (a2 + c))
    return ret.reshape((1, ret.size))


adhTrans[2, 2] = lambda y : S(y)

# setup reactions
R = np.empty(3, dtype=object)

# parameters
K1    = 324. / 7
K2    = 1.
Gamma = 40.  / 7
b     = 1.   / 9
K     = 1.   / 7
r1    = 0.1
r2    = 1.6
r3    = 4.0
mu    = 0.3

# for c(x, t)
R[0] = lambda c, h, u : mu * K1 * h * (b + c) / (1 + c)
# h(x, t)
R[1] = lambda c, h, u : K2**2 / (K2**2 + c**2) - h
# u(x, t)
R[2] = lambda c, h, u : r1 * u * (1 - u) * (1 + r2 * c**2 / (r3 + c**2))

# assemble initial condition
x  = interval.xs()
y0 = np.zeros((3, x.size))

# ICs
css = 0.5563278750155162
# c(0, x)
y0[0, :] = css + 1.42857 * np.exp(-0.25 * (x - 0.5 * L)**2)
# h(0, x)
y0[1, :] = 1. / (1. + css**2)
# u(0, x)
y0[2, :] = np.exp(-0.25 * (x - 0.5 * L)**2)

# simulation time control
time = Time(t0 = 0, tf = 5, dt = 0.1)

# finally setup the solver
solver = MOL(TDR, y0, domain=interval, transitionMatrix=trans, int_mode='morse',
             AdhesionTransitionMatrix=adhTrans, reactionTerms=R,
             time=time, verbose=True, R=Rs)

# save runtime start
runTimeStart = datetime.now()

solver.run()

print('\nSimulation completed!')
print('Execution time  :', datetime.now() - startTime)
print('Simulation time :', datetime.now() - runTimeStart)

