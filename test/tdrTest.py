#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import unittest
import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mol.MOL import MOL
from tdr.TDR import TDR

from tdr.Domain import Interval
from model.Time import Time

from vis.plotPdeSoln1d import plot
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


""" Analytical n(t, r) for the 1d taxis test """
def n(t, r):
    sv = s(t, r)
    init = n0(sv)
    val =  init * (np.sin(4. * np.pi * sv) / (np.sin(4. * np.pi * r) + 1.e-16 * (np.abs(r) < 1.e-16)))
    # Smooth out division problem
    h = (r[1] - r[0])/2.
    mask1 = np.where((r > 0.25 - h)&(r < 0.25 + h))
    val[mask1] = n0(r[mask1]) * np.exp(16. * np.pi * np.pi * t)
    val[0] = init[0] * np.exp(-16. * np.pi * np.pi * t)
    return val


""" Analytical solution for 1d diffusion test with Neumann boundary conditions """
def diffusion_1d_analyticSolution(t, x, k = 1, D = 0.1):
    l = (0.5 * k * np.pi)**2
    soln = 2. + np.exp(-l * t * D) * np.cos(0.5 * k * np.pi * x)
    return soln


def plotInit(x, c0, n0, fname=None):
    yMax = max(np.max(c0), np.max(n0))
    plt.plot(x, c0, label='c0')
    plt.plot(x, n0, label='n0')
    plt.legend(loc='best')
    plt.xlabel('Radial position')
    plt.ylabel('Density')
    plt.xlim([0, 0.5])
    plt.ylim([0, 1.1 * yMax])

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)


def compute_error(times, x, dfs, ansoln, norm = lambda x : np.abs(x),
                  tol=1.e-16, mask=None):
    # We assume that dfs and ansoln have the same keys!!
    error_dfs = {}

    # If we don't have a mask, create a mask that doesn't mask
    if mask is None:
        mask = np.where(x)

    for key in dfs.keys():
        error_dfs[key] = pd.DataFrame()
        for time in times:
            numerical_soln = dfs[key][time].values[mask]
            analyical_soln = np.nan_to_num(ansoln[key](time, x[mask]), copy=True)
            error_dfs[key][time] = np.maximum(tol, norm(numerical_soln - analyical_soln))

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


def plot_soln_diff(time, x, dfs, fname=None):
    edens = diffusion_1d_analyticSolution(time, x)

    plt.clf()
    plt.title('Time %.4g' % time)
    plt.grid()
    ymax = 1.2 * np.max(dfs[0][time].values)
    plt.plot(x, dfs[0][time].values, label='Numerical')
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


class Taxis1dUnitTest(unittest.TestCase):

    def setUp(self):
        # solver tolerance
        self.vtol = 1.e-8

        # reset solver
        self.solver = None

        # 1D
        self.nop = 1

        # norm functionals
        self.norm1 = lambda x : norm1(x, self.interval.h)
        self.norm2 = lambda x : norm2(x, self.interval.h)

        # Make sure these values are resetted
        self.name = 'Unknown'
        self.expected_errors = {}


    def _taxis_setup(self):
        self.L = 0.5
        self.interval = Interval(-self.L, self.L, n = 12)
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

        # functors for analyical solutions
        self.ansoln = {0 : lambda t, x : concentrationField(t, x),
                       1 : lambda t, x : n(t, x)}


    def _diffusion_setup(self):
        self.L = 2.
        self.interval = Interval(0, self.L, n = 12)

        # number of equations
        #
        # Equation 1: is C(x, t)
        # Equation 2: is N(x, t)
        #
        self.fieldNames = ['u(x, t)']
        self.size = 1

        # transition matrices
        self.D           = 0.1
        self.trans       = self.D * np.ones((1,1))

        # functors for analyical solutions
        self.ansoln = {0 : lambda t, x : diffusion_1d_analyticSolution(t, x)}


    def get_ic_taxis(self):
        x = self.interval.xs()
        _c0 = concentrationField(0, x)
        _n0 = n0(x)

        #plotInit(x, c0, n0)

        # assemble the initial condition
        return np.row_stack((_c0, _n0))


    def get_ic_diff(self, k = 1):
        x = self.interval.xs()
        return np.cos(k * np.pi * x / self.L) + 2.


    def exec_solver(self, time, ic_gen):
        # Create solver object
        y0 = ic_gen()

        self.solver = MOL(TDR, y0, nop = self.nop, domain = self.interval,
                          livePlotting=False, noPDEs = self.size,
                          transitionMatrix = self.trans, time = time, vtol=self.vtol)

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


    def do_final_plots(self, plot_fn):
        times = self.get_times()
        x = self.interval.xs()
        dfs = self.solver.dfs

        # to delete them after movie creation
        files = []

        idx = 0
        for time in times:
            fname = os.path.join(self.outdir, self.name + '_soln_' + str(idx).zfill(4) + '.png')
            files.append(fname)
            idx += 1
            plot_fn(time, x, dfs, fname = fname)

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

        # get data frame
        pos = self.get_position()

        efname = os.path.join(self.outdir, self.name + '_erro.png')
        plot(self.error_dfs,    position=pos, names = self.fieldNames, fname = efname, vmin = 1.e-16)


    def plot_soln(self, mask = None):
        # get data frame
        pos = self.get_position()
        sfname = os.path.join(self.outdir, self.name + '_soln.png')

        dfs = copy.deepcopy(self.solver.dfs)
        if mask is not None:
            dfs = self.mask_dfs(mask)

        plot(dfs,   position=pos, names = self.fieldNames, fname = sfname)


    def get_times(self):
        dfs = self.solver.dfs
        # assume that all the times are the same
        return dfs[0].columns.values


    def get_error(self, mask=None):
        x = self.interval.xs()
        times = self.get_times()
        error_dfs = compute_error(times, x, self.solver.dfs, self.ansoln,mask=mask)
        return error_dfs


    def verify_error(self, mask = None):
        self.do_error_plots(mask=mask)
        errors = self.compute_error_norm()
        for key, error in errors.items():
            err = np.max(error)
            self.assertLessEqual(err, self.expected_errors[key])


    def get_position(self):
        return np.array([0, self.L])


    def test_taxis1d_short_time(self):
        self._taxis_setup()

        # set name
        self.name = 'taxis1d_short_time_test'
        self.expected_errors = {0 : 1e-16, 1 : 1e-3}

        # check outdir
        self.check_outdir()

        # time
        time = Time(t0 = 0., tf = 0.007, dt = 0.0001)

        # execute the solver
        self.exec_solver(time, self.get_ic_taxis)

        # Only plot the right hand side of the solution interval
        x = self.interval.xs()
        mask = np.where(x >= 0.)

        # PLOT solutions
        self.plot_soln(mask)
        self.do_final_plots(plot_soln)

        # do the actual test that we are accurate enough
        self.verify_error(mask)


    def test_taxis1d_medium_time(self):
        self._taxis_setup()

        # set name
        self.name = 'taxis1d_medium_time_test'
        self.expected_errors = {0 : 1e-16, 1 : 1e-3}

        # check outdir
        self.check_outdir()

        # time
        time = Time(t0 = 0., tf = 0.014, dt = 0.0001)

        # execute the solver
        self.exec_solver(time, self.get_ic_taxis)

        # Only plot the right hand side of the solution interval
        x = self.interval.xs()
        mask = np.where(x >= 0.)

        # PLOT solutions
        self.plot_soln(mask)
        self.do_final_plots(plot_soln)

        # do the actual test that we are accurate enough
        self.verify_error(mask)


    def test_taxis1d_long_time(self):
        self._taxis_setup()

        # set name
        self.name = 'taxis1d_long_time_test'
        self.expected_errors = {0 : 1e-16, 1 : 1e-3}

        # check outdir
        self.check_outdir()

        # time
        time = Time(t0 = 0., tf = 0.021, dt = 0.0001)

        # execute the solver
        self.exec_solver(time, self.get_ic_taxis)

        # Only plot the right hand side of the solution interval
        x = self.interval.xs()
        mask = np.where(x >= 0.)

        # PLOT solutions
        self.plot_soln(mask)
        self.do_final_plots(plot_soln)

        # do the actual test that we are accurate enough
        self.verify_error(mask)


    def test_diffusion1d_neumann(self):
        self._diffusion_setup()

        # set name
        self.name = 'diffusion1d_neumann_test'
        self.expected_errors = {0 : 1e-4}

        # check outdir
        self.check_outdir()

        # time
        time = Time(t0 = 0., tf = 2., dt = 0.01)

        # execute the solver
        self.exec_solver(time, self.get_ic_diff)

        # PLOT solutions
        self.plot_soln()
        self.do_final_plots(plot_soln_diff)

        # do the actual test that we are accurate enough
        self.verify_error()


if __name__ == '__main__':
    unittest.main()
