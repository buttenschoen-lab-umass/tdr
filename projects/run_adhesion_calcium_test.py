#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

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
sys.path.append('/home/adrs0061/sources/NumericalPDEs')

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
from vis.adhesion_new import plot_figure

# animation code
from vis.adhesion_new import create_animation, create_animation_double
from vis.animation import display_animation, save_animation

# setup simulation specific paths!
basename ='AdhesionCalcium-Test-New'
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
    # assemble initial condition
    interval = setup_domain(**pars)
    x = interval.xs()
    y0 = np.zeros((2, x.size))

    # get the st st
    css, hss = compute_st(pars)

    # ICs
    # c(0, x)
    y0[0, :] = css + 1.42857 * np.exp(-0.25 * (x - 0.5 * pars.L)**2)
    # h(0, x)
    y0[1, :] = hss

    # simulation time control
    time = Time(t0 = 0, tf = 10, dt = 0.01)

    # get the reaction terms
    R = create_rxn_term(pars)

    # save start-time
    startTime = datetime.now()

    # set mkl threads
    set_num_threads(kwargs.pop('threads', 28))

    # finally setup the solver
    solver = MOL(TDR, y0, domain=interval, transitionMatrix=pars.trans,
                 reactionTerms=R, outdir=datapath, name=sim_name(mu),
                 vtol=1e-12, ktol=1e-12,
                 time=time, verbose=True, save_new=True)

    # save runtime start
    runTimeStart = datetime.now()

    solver.run()

    print('\nSimulation completed!')
    print('Execution time  :', datetime.now() - startTime)
    print('Simulation time :', datetime.now() - runTimeStart)

    return solver.outfile, interval


def plot(datafile, pars, interval):
    mu = pars['mu']
    fn  = os.path.join(datapath, sim_name(mu)+'.h5')
    ofn = os.path.join(datapath, sim_name(mu)+'_kymo.png')
    plot_figure(fn, interval, outputname=ofn)


def movie(datafile, pars, vars):
    mu = pars['mu']
    L  = pars['L']
    for c in [0, 1]:
        ofn = os.path.join(datapath, sim_name(mu)+'_movie_{0}.avi'.format(vars[c]))
        anim = create_animation(datafile, L, col=c, basename=basename + ' ' + vars[c])
        save_animation(anim, ofn)

    ofn = os.path.join(datapath, sim_name(mu)+'_movie_{0}.avi'.format('double'))
    anim = create_animation_double(datafile, L, basename=basename)
    save_animation(anim, ofn)

if __name__ == '__main__':
    # run this for a few different mus
    mus = np.asarray([0.2, 0.288, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6])

    # define the simulations variables
    vars = {0 : 'c', 1 : 'h'}

    # define the simulations nonlinear functions
    rhs  = {'c' : 'mu * h * K1 * (b + c) / (1. + c) - Gamma * c / (K + c)',
            'h' : 'K2**2 / (K2**2 + c**2) - h'}

    for mu in mus:
        # create simulation parameter object
        pars = Parameters(vars=vars, rhs=rhs, mu=mu, Dc=0.1)

        # run the simulation
        ofile, interval = run_sim(pars)

        # create plots
        plot(ofile, pars, interval)

        # create movies
        movie(ofile, pars, vars)
