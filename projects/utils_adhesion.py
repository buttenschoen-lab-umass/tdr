#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import absolute_import, print_function, division

import os

# set MKL THREADS - We will run 3 processes below -> 3 * 8 = 24
os.environ['MKL_VERBOSE'] = '0'
os.environ['MKL_NUM_THREADS'] = '8'

from utils.threads_mkl import set_num_threads

import numpy as np
import multiprocessing_on_dill as mp
from collections import OrderedDict

from mol.MOL import MOL
from tdr.TDR import TDR
from tdr.Boundary import DomainBoundary, Periodic
from tdr.Domain import Interval
from model.Time import Time
from python.utils import Average


""" Easy creation of spike like initial conditions """
def get_spike_ic(x, n, L, eps=2.5):
    center = lambda k : k * L / n

    y = np.zeros_like(x)
    for k in range(n):
        c = center(k)

        # the first spike will be on the boundary -> so flip around so that we get both pieces
        if k == 0:
            yt = 1./np.cosh(eps*(x - c))
            y += yt + yt[::-1]
        else:
            y += 1./np.cosh(eps*(x - c))

    return y


""" Factory method for a MOL solver for non-local adhesion problems """
def create_solver(alpha, L, t0 = 0, tf = 5, no_data_points = 100,
                  randomInit=True, outdir=None, n=8, nonLocalMode='periodic',
                  verbose=False, k = 1, int_mode = 'uniform', threads=12,
                  save=False, vtol=1e-6, basename='AdhesionModelPeriodicStStInvestigation',
                  bdType=Periodic, *args, **kwargs):
    if outdir is None:
        print('WARNING: outdir is not SET! Not saving anything!')

    print('Creating solver with alpha = %.2f.' % alpha)
    bd = DomainBoundary(left=bdType(-1.), right=bdType(1.))
    interval = Interval(0, L, n=n, bd=bd)

    x = interval.xs()
    scale = kwargs.pop('scale', 0.01)
    loc   = kwargs.pop('loc', 0.01)
    ic    = kwargs.pop('ic', 'noise')
    u0    = kwargs.pop('u0', 1.)

    if ic == 'noise':
        print('Creating IC using Gaussian noise (%.4g, %.4g).' % (loc, scale))
        y0 = u0 + np.random.normal(loc=loc, scale=scale, size=x.size)
    elif ic == 'spike':
        print('Creating IC using spikes (%.4g, %.4g).' % (loc, L))
        y0 = scale * get_spike_ic(x, loc, L)
        canonical_tile = L / (2. * loc)
        y0 = np.roll(y0, int(canonical_tile * interval.N/L))
    else:
        print('Creating IC using sine perturbation (%.4g, %.4g).' % (loc, scale))
        y0 = u0 + scale * np.sin(loc * k * np.pi * x / L)

    M = Average(interval.xs(), y0)
    y0 /= M
    y0 *= u0

    print('Average of y0 is: %.4g.' % Average(interval.xs(), y0))

    # transition matrices
    trans    = np.ones(1).reshape((1,1))
    Adhtrans = alpha * np.ones(1).reshape((1,1))

    # times
    print(tf, t0, no_data_points)
    dt = max((tf - t0) / no_data_points, 0.01)
    time = Time(t0 = t0, tf = tf, dt = dt)

    name = basename + '_L=%f_alpha=%f_tf=%f' % (L, alpha, tf)
    print('Saving to: %s.' % name)
    print('outdir: %s.' % outdir)

    # setting mkl threads
    print('Setting MKL threads to %d.' % (threads))
    set_num_threads(threads)

    solver = MOL(TDR, y0, domain=interval, transitionMatrix=trans,
                 nonLocalMode=nonLocalMode, AdhesionTransitionMatrix=Adhtrans,
                 time=time, vtol=vtol, name=name, outdir=outdir, save=save,
                 verbose=verbose, int_mode=int_mode, max_iter=100000,
                 *args, **kwargs)

    return solver, interval, y0


""" Functions to easy run several simulations """
def run_simulation(alpha, L, tf, datapath, basename, *args, **kwargs):
    solver, interval, y0 = create_solver(alpha, L, tf=tf, outdir=datapath,
                                         basename=basename, *args, **kwargs)
    solver.run()
    return solver.outfile, interval


def run_simulations(alphas, L, tf, datapath, basename, ns, *args, **kwargs):
    data = OrderedDict()
    intervals = {}
    for idx, alpha in enumerate(alphas):
        dfs, interval = run_simulation(alpha, L, tf, datapath, basename, loc=ns[idx], *args, **kwargs)
        data[alpha] = dfs
        intervals[alpha] = interval

    return data, intervals


def run_simulations_mp(alphas, L, tf, datapath, basename, ns, *args, **kwargs):
    data = OrderedDict()
    pool = mp.Pool(min(len(alphas), 4))

    def run_sim(idx):
        alpha = alphas[idx]
        loc   = ns[idx]
        datafile, interval = run_simulation(alpha, L, tf, datapath, basename, loc=loc, *args, **kwargs)
        return alpha, datafile

    idxs = range(len(alphas))
    result_map  = pool.map(run_sim, idxs)

    for alpha, sdata in (r for r in result_map if r is not None):
        data[alpha]      = sdata

    pool.close()
    pool.join()

    return data
