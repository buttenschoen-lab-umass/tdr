#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import os, sys

# set MKL THREADS - We will run 3 processes below -> 3 * 8 = 24
os.environ['MKL_VERBOSE'] = '0'
os.environ['MKL_NUM_THREADS'] = '28'

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
from vis.adhesion_new import plot_figure

# animation code
from vis.adhesion_new import create_animation
from vis.animation import display_animation, save_animation

# setup simulation specific paths!
basename ='AdhesionCalcium-Population-New-3-Test'
datapath = os.path.join(basepath, 'results', 'adhesion-calcium', basename)

# make sure path we will use exists!
check_path(datapath)

# We have three concentrations
#
#   - c(x, t): Calcium as a function of space and time
#   - h(x, t): ??
#
def sim_name(mu):
    return basename + '_mu=%d' % mu

def run_sim(pars, idx, *args, **kwargs):
    # save start-time
    startTime = datetime.now()

    # assemble initial condition
    interval = setup_domain(**pars)
    x = interval.xs()
    y0 = np.zeros((1, x.size))

    # ICs
    # u(0, x)
    y0[0, :] = np.exp(-0.25 * (x - 0.5 * pars.L)**2)

    # simulation time control
    time = Time(t0 = 0, tf = 500, dt = 0.25)

    # get the reaction terms
    R = create_rxn_term(pars)

    # This function is a ToDO!
    adhTrans = np.zeros((1, 1))
    adhTrans[0, 0] = pars.s_star

    # assemble diffusion matrix
    trans = np.zeros((1, 1))
    trans[0, 0] = pars.Du

    # set mkl threads
    set_num_threads(kwargs.pop('threads', 28))

    # finally setup the solver
    solver = MOL(TDR, y0, domain=interval, transitionMatrix=trans,
                 int_mode='morse', AdhesionTransitionMatrix=adhTrans,
                 reactionTerms=R, outdir=datapath, name=sim_name(idx),
                 vtol=1e-7, ktol=1e-1, R=pars.Rs, qa=pars.qa, qr=pars.qr,
                 sr=pars.sr, sa=pars.sa, mr=pars.mr, ma=pars.ma,
                 time=time, verbose=True, save_new=True)

    # save runtime start
    runTimeStart = datetime.now()

    solver.run()

    print('\nSimulation completed!')
    print('Execution time  :', datetime.now() - startTime)
    print('Simulation time :', datetime.now() - runTimeStart)

    return solver.outfile, interval


def plot(datafile, pars, interval, idx):
    fn  = os.path.join(datapath, sim_name(idx)+'.h5')
    ofn = os.path.join(datapath, sim_name(idx)+'_kymo.png')
    plot_figure(fn, interval, outputname=ofn)


def movie(datafile, pars, vars, idx):
    L  = pars['L']
    ofn = os.path.join(datapath, sim_name(idx)+'_movie_{0}.avi'.format(vars[0]))
    anim = create_animation(datafile, L, col=0, basename=basename + ' ' + vars[0])
    save_animation(anim, ofn)


if __name__ == '__main__':
    # run this for a few different mus
    pars = [{'a1' : 0, 'a2' : 0, 'r1' : 0, 'r2' : 0, 'qa' : 0.22, 'qr' : 0.01},
            {'qa' : 0, 'qr' : 0},
            {'qa' : 0.22, 'qr' : 0.01},
            {'qa' : 0.09, 'qr' : 0.22},
            {'qa' : 0.22, 'qr' : 0.22},
            {'qa' : 0.09, 'qr' : 0.01}]

    # define the simulations variables
    vars = {0 : 'u'}

    # define the simulations nonlinear functions
    rhs  = {'u' : 'r1 * u * (1 - u)'}

    for idx, par in enumerate(pars):
        # create simulation parameter object
        pars = Parameters(vars=vars, rhs=rhs, Du=0.0025, **par)

        # run the simulation
        ofile, interval = run_sim(pars, idx)

        # create plots
        plot(ofile, pars, interval, idx)

        # create movies
        movie(ofile, pars, vars, idx)
