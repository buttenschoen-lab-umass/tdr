#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author: Andreas Buttenschoen
#
from __future__ import print_function, division, absolute_import

import unittest
import os
import copy

# set MKL THREADS
os.environ['MKL_VERBOSE'] = '0'
os.environ['MKL_NUM_THREADS'] = '14'

from utils.threads_mkl import set_num_threads

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm

from mol.MOL import MOL
from tdr.TDR import TDR

from tdr.Domain import Square
from model.Time import Time

from python.utils import norm1, norm2

# Simple ffmpeg movie support
from im2movie import makeMovie


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


""" Analytical s(t, r) for the 1d taxis test """
def s(t, r):
    #int_part = np.asarray(4. * r, dtype = np.int)
    #int_mod  = np.mod(int_part, 2)
    val = np.arctan(np.tan(2. * np.pi * r) / np.exp(16. * np.pi * np.pi * t))
    val /= (2. * np.pi)
    mask = np.where(r > 0.25)
    val[mask] += 0.5 * np.ones_like(mask[0])
    #val += 0.25 * (int_part + int_mod)
    return val


""" Analytical n(t, r) for the 2d taxis test """
def n(t, r):
    sv = s(t, r)
    init = n0(sv)
    val =  init * (sv / r) * (np.sin(4. * np.pi * sv) / np.sin(4. * np.pi * r))
    # Smooth out division problem
    h = (r[1] - r[0])/2.
    mask1 = np.where((r > 0.25 - h)&(r < 0.25 + h))
    val[mask1] = n0(r[mask1]) * np.exp(16. * np.pi * np.pi * t)
    val[0] = init[0] * np.exp(-32. * np.pi * np.pi * t)
    return val


def plotInit(xx, yy, c0, n0, fname=None):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title('$C(x,0)$')
    plt.pcolormesh(xx, yy, c0)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.title('$N(x,0)$')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.pcolormesh(xx, yy, n0)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.colorbar()
    plt.tight_layout()

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

    plt.close()


def plot(dfs, xx, yy, time = None, names = None, fname=None, tround=4, log=False, *args, **kwargs):
    nx = 1
    ny = len(dfs)

    bounds = kwargs.pop('bounds', { k : {'ymin' : None, 'ymax' : None} for k in dfs.keys()})
    plt.figure(figsize=(20,10))

    # create meshgrid
    x, y = np.meshgrid(xx, yy)

    for i, (key, df) in enumerate(dfs.items()):
        plt.subplot(nx, ny, i+1)
        ax = plt.gca()
        ax.set_aspect('equal')

        # make sure indices are rounded so that we can guarantee lookup success
        df = df.set_index(np.round(df.index, decimals=tround))

        if time is None:
            time = df.index[-1]

        if names is not None:
            plt.title(names[i] + ' @ %.5f' % time)

        # extract the values we need
        z  = df.loc[time].values
        sz = np.sqrt(z.size).astype(np.int)
        z  = z.reshape((sz,sz))

        bound = bounds[key]
        norm = None
        if log:
            vmin = kwargs.pop('vmin', 1e-16)
            vmax = kwargs.pop('vmax', 1e-1)
            norm = LogNorm(vmin=min(vmin, bound['ymin']), vmax=max(vmax, bound['ymax']))
        else:
            norm = Normalize(vmin=bound['ymin'], vmax=bound['ymax'])
        sm   = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        sm._A = []
        plt.pcolormesh(x, y, z, norm=norm, cmap=cm.viridis, *args, **kwargs)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar(sm)

    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

    plt.close()


def compute_error(times, x, y, dfs, ansoln, norm = lambda x : np.abs(x),
                  tol=1.e-16, mask=None, tround=4):
    # We assume that dfs and ansoln have the same keys!!
    error_dfs = {}

    # If we don't have a mask, create a mask that doesn't mask
    if mask is None:
        mask = np.where(x)

    for key in dfs.keys():
        error_dfs[key] = pd.DataFrame()
        df = dfs[key]
        # make sure index allows for float access
        df = df.set_index(np.round(df.index, decimals=tround))

        for time in times:
            numerical_soln = df.loc[time].values
            sz = np.sqrt(numerical_soln.size).astype(np.int)
            numerical_soln = numerical_soln.reshape((sz, sz))
            xx, yy = np.meshgrid(x, y, sparse=True)
            analyical_soln = np.nan_to_num(ansoln[key](time, np.sqrt(xx**2 + yy**2)), copy=True)
            error_dfs[key][time] = np.maximum(tol, norm(numerical_soln - analyical_soln)).flatten()

        # FIXME properly
        error_dfs[key] = error_dfs[key].transpose()

    return error_dfs


def plot_soln(time, x, dfs, fname=None):
    mask  = np.where(x >= 0)
    mask2 = np.where(x < 0)
    edens = np.zeros_like(x)
    edens[mask]  = n(time, x[mask])
    edens[mask2] = np.flip(edens[mask], 0)

    plt.clf()
    plt.title('Time %.4g' % time)
    plt.grid()
    ymax = 1.2 * np.max(dfs[1][time].values)
    plt.plot(x, dfs[1][time].values, label='Numerical')
    plt.plot(x, edens, label='Analytical', ls='--')
    plt.ylabel('Density')
    plt.xlabel('x')
    plt.legend(loc='best')
    plt.xlim([0., np.max(x)])
    plt.ylim([0, ymax])

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

    plt.close()


#def plot_soln_diff(time, x, dfs, fname=None):
#    edens = diffusion_1d_analyticSolution(time, x)
#
#    plt.clf()
#    plt.title('Time %.4g' % time)
#    plt.grid()
#    ymax = 1.2 * np.max(dfs[0][time].values)
#    plt.plot(x, dfs[0][time].values, label='Numerical')
#    plt.plot(x, edens, label='Analytical', ls='--')
#    plt.ylabel('Density')
#    plt.xlabel('x')
#    plt.legend(loc='best')
#    plt.xlim([0., np.max(x)])
#    plt.ylim([0, ymax])
#
#    if fname is None:
#        plt.show()
#    else:
#        plt.savefig(fname)


class tdrTest2D(unittest.TestCase):

    def setUp(self):
        # solver tolerance
        self.vtol = 1.e-8

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

        # functors for analytical solutions
        self.ansoln = {0 : lambda t, r : concentrationField(t, r),
                       1 : lambda t, r : n(t, r)}


    #def _diffusion_setup(self):
    #    self.L = 2.
    #    self.time_round = 2
    #    self.domain = Square(0, self.L, 0, self.L, n = 10)

    #    # number of equations
    #    #
    #    # Equation 1: is C(x, t)
    #    # Equation 2: is N(x, t)
    #    #
    #    self.fieldNames = ['u(x, t)']
    #    self.size = 1

    #    # transition matrices
    #    self.D           = 0.1
    #    self.trans       = self.D * np.ones((1,1))

    #    # functors for analytical solutions
    #    self.ansoln = {0 : lambda t, x : diffusion_1d_analyticSolution(t, x)}


    def get_ic_taxis(self):
        # create a coordinate grid
        xs = self.domain.xs()
        ys = self.domain.ys()
        xx, yy = np.meshgrid(xs, ys, sparse=True)

        _c0 = concentrationField(0, np.sqrt(xx**2 + yy**2))
        _n0 = n0(np.sqrt(xx**2 + yy**2))

        fname = os.path.join(self.outdir, self.name + '_ic.png')
        plotInit(xx, yy, _c0, _n0, fname=fname)

        # reshape
        _c0 = np.expand_dims(_c0, axis=0)
        _n0 = np.expand_dims(_n0, axis=0)

        # assemble the initial condition
        return np.row_stack((_c0, _n0))


    def get_ic_diff(self, k = 1):
        x = self.interval.xs()
        return np.cos(k * np.pi * x / self.L) + 2.


    def exec_solver(self, time, ic_gen):
        # Create solver object
        y0 = ic_gen()

        # since we might be using mkl use threads
        set_num_threads(self.threads)

        self.solver = MOL(TDR, y0, nop = self.nop, domain = self.domain,
                          livePlotting=False, transitionMatrix = self.trans,
                          time = time, vtol=self.vtol)

        self.solver.run()


    def mask_dfs(self, mask):
        dfs = self.solver.dfs
        ndfs = {}
        for key, df in dfs.items():
            cols = df.columns.values
            ndfs[key] = pd.DataFrame()
            ndf = ndfs[key]
            for col in cols:
                ndf[col] = df[col].values[mask]
        return ndfs


    def check_outdir(self):
        cwd = os.getcwd()
        self.outdir = os.path.join(cwd, 'results', 'tests', self.name)
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)


    def get_ybounds(self, dfs):
        bounds = {}
        for key, df in dfs.items():
            bounds[key] = {}
            bounds[key]['ymax'] = np.max(df.values)
            bounds[key]['ymin'] = np.min(df.values)

        return bounds


    def do_final_plots(self, plot_fn):
        times = self.get_times()
        xs = self.domain.xs()
        ys = self.domain.ys()
        dfs = self.solver.dfs

        # get bounds
        bounds = self.get_ybounds(dfs)

        # to delete them after movie creation
        files = []

        idx = 0
        for time in times:
            fname = os.path.join(self.outdir, self.name + '_soln_' + str(idx).zfill(4) + '.png')
            files.append(fname)
            idx += 1
            plot_fn(dfs, xs, ys, time = time, names = self.fieldNames, fname = fname, bounds=bounds)

        movie_name          = self.name + '_soln'
        makeMovie(movie_name,'.png',movie_name,self.outdir,self.outdir,20,quiet=True)

        # remove files
        for fn in files:
            try:
                os.remove(fn)
            except OSError:
                print('ERROR encountered!!')


    def compute_error_norm(self, norm = None):
        if norm is None:
            norm = self.norm1

        errors = {}
        for key, df in self.error_dfs.items():
            cols = df.columns.values
            errors[key] = []
            for col in cols:
                errors[key].append(norm(df[col].values))

        # return the worst case scenario
        return errors


    def do_error_plots(self, mask=None):
        # compute the error
        self.error_dfs = self.get_error(mask=mask)

        # get coordinates
        xs = self.domain.xs()
        ys = self.domain.ys()

        efname = os.path.join(self.outdir, self.name + '_erro.png')

        bounds = self.get_ybounds(self.error_dfs)
        plot(self.error_dfs, xs, ys, names = self.fieldNames, fname = efname,
             log = True, vmin = 1.e-16, bounds=bounds)


    def plot_soln(self, mask = None):
        # get data frame
        #pos = self.get_position()
        sfname = os.path.join(self.outdir, self.name + '_soln.png')

        dfs = copy.deepcopy(self.solver.dfs)
        if mask is not None:
            dfs = self.mask_dfs(mask)

        xs = self.domain.xs()
        ys = self.domain.ys()

        plot(dfs, xs, ys, names = self.fieldNames, fname = sfname)


    def get_times(self):
        dfs = self.solver.dfs
        # assume that all the times are the same
        return np.round(dfs[0].transpose().columns.values,
                        decimals=self.time_round)


    def get_error(self, mask=None):
        xs    = self.domain.xs()
        ys    = self.domain.ys()
        times = self.get_times()
        error_dfs = compute_error(times, xs, ys, self.solver.dfs, self.ansoln,mask=mask)
        return error_dfs


    def verify_error(self, mask = None):
        self.do_error_plots(mask=mask)
        errors = self.compute_error_norm()
        for key, error in errors.items():
            err = np.max(error)
            self.assertLessEqual(err, self.expected_errors[key])


    def get_position(self):
        return np.array([lambda x : 0, lambda x : self.L, lambda x : 0, lambda x : self.L])


    def test_taxis_short_time(self):
        print('setup taxis test.')
        self._taxis_setup()

        # set name
        self.name = 'taxis2d_short_time_test'
        self.expected_errors = {0 : 1e-16, 1 : 1e-3}

        # check outdir
        self.check_outdir()

        # time
        time = Time(t0 = 0., tf = 0.007, dt = 0.0001)

        # execute the solver
        print('execute the solver')
        self.exec_solver(time, self.get_ic_taxis)

        # PLOT solutions
        self.plot_soln()
        self.do_final_plots(plot)

        # do the actual test that we are accurate enough
        self.verify_error()


    def test_taxis_medium_time(self):
        self._taxis_setup()

        # set name
        self.name = 'taxis2d_medium_time_test'
        self.expected_errors = {0 : 1e-16, 1 : 1e-3}

        # check outdir
        self.check_outdir()

        # time
        time = Time(t0 = 0., tf = 0.014, dt = 0.0001)

        # execute the solver
        self.exec_solver(time, self.get_ic_taxis)

        # PLOT solutions
        self.plot_soln()
        self.do_final_plots(plot)

        # do the actual test that we are accurate enough
        self.verify_error()


    def test_taxis_long_time(self):
        self._taxis_setup()

        # set name
        self.name = 'taxis2d_long_time_test'
        self.expected_errors = {0 : 1e-16, 1 : 1e-3}

        # check outdir
        self.check_outdir()

        # time
        time = Time(t0 = 0., tf = 0.021, dt = 0.0001)

        # execute the solver
        self.exec_solver(time, self.get_ic_taxis)

        # PLOT solutions
        self.plot_soln()
        self.do_final_plots(plot)

        # do the actual test that we are accurate enough
        self.verify_error()


    #def test_diffusion_neumann(self):
    #    self._diffusion_setup()

    #    # set name
    #    self.name = 'diffusion2d_neumann_test'
    #    self.expected_errors = {0 : 1e-4}

    #    # check outdir
    #    self.check_outdir()

    #    # time
    #    time = Time(t0 = 0., tf = 2., dt = 0.01)

    #    # execute the solver
    #    self.exec_solver(time, self.get_ic_diff)

    #    # PLOT solutions
    #    self.plot_soln()
    #    self.do_final_plots(plot_soln_diff)

    #    # do the actual test that we are accurate enough
    #    self.verify_error()


if __name__ == '__main__':
    unittest.main()
