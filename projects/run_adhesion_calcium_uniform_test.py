#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import absolute_import, print_function, division

import os, sys

# set MKL THREADS - We will run 3 processes below -> 3 * 8 = 24
os.environ['MKL_VERBOSE'] = '0'
os.environ['MKL_NUM_THREADS'] = '8'

from datetime import datetime
import numpy as np
import numba
from numba import jit

# Setup the paths!
homedir  = os.path.expanduser('~')
basepath = os.path.join(homedir,  'sources', 'NumericalPDEs')

# append system path
sys.path.append(basepath)

from utils.threads_mkl import set_num_threads

# import calcium utilities
from projects.utils_calcium import check_path, create_rxn_term, setup_domain, Parameters
from projects.utils_calcium import compute_st

from mol.MOL import MOL
from tdr.Domain import Interval
from tdr.Boundary import DomainBoundary, Periodic
from tdr.TDR import TDR
from model.Time import Time

# import plotting code
from vis.adhesion_new import plot_osc

# setup simulation specific paths!
basename ='AdhesionCalcium-Uniform-Test'
datapath = os.path.join(basepath, 'results', 'adhesion-calcium', basename)

# make sure path we will use exists!
check_path(datapath)

# We have three concentrations
#
#   - c(x, t): Calcium as a function of space and time
#   - h(x, t): ??
#
def sim_name(mu):
    return basename + '_mu=%.2f' % mu

def run_sim(pars, *args, **kwargs):
    interval = setup_domain(**pars)
    x  = interval.xs()

    # Create initial condition
    y0 = np.zeros((2, x.size))

    # compute steady states
    css, hss = compute_st(pars)

    # uniform steady state
    y0[0, :] = css * np.ones_like(x)
    # h(0, x)
    y0[1, :] = hss * np.ones_like(x)

    # simulation time control
    time = Time(t0 = 0, tf = 30, dt = 0.05)

    # get the reaction terms
    R = create_rxn_term(pars)

    # save start-time
    startTime = datetime.now()

    # set mkl threads
    set_num_threads(kwargs.pop('threads', 28))

    # finally setup the solver
    solver = MOL(TDR, y0, domain=interval, transitionMatrix=pars.trans,
                 reactionTerms=R, outdir=datapath, name=sim_name(mu),
                 time=time, verbose=True, save_new=True)

    # save runtime start
    runTimeStart = datetime.now()

    # Start simulation
    solver.run()

    print('\nSimulation completed!')
    print('Execution time  :', datetime.now() - startTime)
    print('Simulation time :', datetime.now() - runTimeStart)

    # return the datafile path to which we just wrote
    return solver.outfile


def plot(datafile, pars):
    mu = pars['mu']
    fn  = os.path.join(datapath, sim_name(mu)+'.h5')
    ofn = os.path.join(datapath, sim_name(mu)+'_amplitude.png')
    plot_osc(fn, outputname=ofn, titles=['Calcium $\\mu = %.2f$' % mu, 'h $\\mu = %.2f$' % mu],
             xmin=10, xmax=30, ymax=5.5)


if __name__ == '__main__':
    # run this for a few different mus
    mus = np.asarray([0.2, 0.25, 0.288, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6])

    # define the simulations variables
    vars = {0 : 'c', 1 : 'h'}

    # define the simulations nonlinear functions
    rhs  = {'c' : 'mu * h * K1 * (b + c) / (1. + c) - Gamma * c / (K + c)',
            'h' : 'K2**2 / (K2**2 + c**2) - h'}

    # run the simulations for each of the mu - values
    for mu in mus:
        # create simulation parameter object
        pars = Parameters(vars=vars, rhs=rhs, mu = mu)

        # run the simulation
        ofile = run_sim(pars)

        # create plots
        plot(ofile, pars)
