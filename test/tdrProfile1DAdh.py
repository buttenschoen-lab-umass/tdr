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

from projects.utils_adhesion import run_simulation


if __name__ == '__main__':
    alpha = 10.

    homedir  = os.path.expanduser('~')
    basepath = os.path.join(homedir,  'sources', 'NumericalPDEs')
    basename ='AdhesionStStProfile'
    datapath = os.path.join(basepath, 'results', 'adhesion', basename)

    if not os.path.exists(datapath):
        os.makedirs(datapath)

    # Domain information
    L = 3
    n = 2

    # Domain specific example
    a1 = 1./128
    a2 = 1./512
    r1 = 0.05
    r2 = L / (2*n) + 0.1

    # simulation data
    tf    = 10.
    u0    = 10.

    data, interval = run_simulation(alpha, L, tf, datapath, basename, scale=0.025,
                              no_data_points=5000, u0=u0, ic='spike', verbose=True, loc=4,
                              int_mode='double-peak', a1=a1, a2=a2, r1=r1, r2=r2, n=10)

