#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.ndimage

from mol.MOL import MOL
from tdr.TDR import TDR
from tdr.Boundary import DomainBoundary
from tdr.Boundary import Periodic

from tdr.Domain import Square
from model.Time import Time


def r(x, y):
    return np.sqrt((x - 0.5)**2 + (y - 0.5)**2)


def c1(x, y):
    return 1. - np.cos(4. * np.pi * r(x, y))


def n0_polar(r, kappa = 0.1):
    nInit = np.zeros_like(r)
    mask1 = np.where(np.abs(r) <= 0.4 - kappa)
    mask2 = np.where((0.4 - kappa <= np.abs(r))&(np.abs(r) <= 0.4 + kappa))
    mask3 = np.where(np.abs(r) > 0.4 + kappa)
    nInit[mask1] = 1.
    nInit[mask2] = 0.5 * (1. + np.cos(np.pi * (np.abs(r[mask2]) - 0.4 + kappa)/(2. * kappa)))
    nInit[mask3] = 0.
    return nInit


def n0(x, y):
    return n0_polar(r(x, y))


def s(t, r):
    #int_part = np.asarray(4. * r, dtype = np.int)
    #int_mod  = np.mod(int_part, 2)
    val = np.arctan(np.tan(2. * np.pi * r) / np.exp(16. * np.pi * np.pi * t))
    val /= (2. * np.pi)
    mask = np.where(r > 0.25)
    val[mask] += 0.5 * np.ones_like(mask[0])
    #val += 0.25 * (int_part + int_mod)
    return val


def n_polar(t, r):
    sv = s(t, r)
    init = n0_polar(sv)
    val =  init * (sv / r) * (np.sin(4. * np.pi * sv) / np.sin(4. * np.pi * r))
    # Smooth out division problem
    h = (r[1] - r[0])/2.
    mask1 = np.where((r > 0.25 - h)&(r < 0.25 + h))
    val[mask1] = n0_polar(r[mask1]) * np.exp(16. * np.pi * np.pi * t)
    val[0] = init[0] * np.exp(-32. * np.pi * np.pi * t)
    return val


def nc(t, x, y):
    return n_polar(t, r(x, y))


def radial_profile(data, center = None):
    y, x = np.indices((data.shape))
    if center is None:
        center = np.asarray([np.max(y) / 2, np.max(x) / 2])

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


if __name__ == '__main__':
    #L = 0.5

    # time
    time = Time(t0 = 0., tf = 0.014, dt = 0.001)

    # C0 = c1(X, Y)

    #plt.pcolormesh(X, Y, Z)
    #plt.colorbar()
    #plt.show()

    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, C0, rstride=1, cstride=1, cmap='viridis',
    #                 edgecolor='none')
    # ax.set_title('surface')
    # plt.show()

    # initial condition
    # N0 = n0(X, Y)

    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, N0, rstride=1, cstride=1, cmap='viridis',
    #                 edgecolor='none')
    # ax.set_title('surface')
    # plt.show()

    # create domain
    n = 9
    h = 2**(-n)
    L = 1 + h
    N = 2**n + 1
    square = Square(0, L, 0, L, n = n)

    x = square.xs()
    y = square.ys()

    X, Y = np.meshgrid(x, y)

    C0 = c1(X, Y)
    N0 = n0(X, Y)

    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1, aspect=1)
    plt.title('C0')
    plt.pcolormesh(X, Y, C0)
    plt.colorbar()

    plt.subplot(1,2,2, aspect=1)
    plt.title('N0')
    plt.pcolormesh(X, Y, N0)
    plt.colorbar()
    plt.show()

    C0 = np.expand_dims(c1(X, Y), axis=0)
    N0 = np.expand_dims(n0(X, Y), axis=0)

    # number of equations
    #
    # Equation 1: is C(x, t)
    # Equation 2: is N(x, t)
    #
    fieldNames = ['C(x, t)', 'N(x, t)']
    size = 2

    # 1D
    nop = 1

    # transition matrices
    trans       = np.zeros((2,2))

    # set one taxis coefficient to one
    trans[1, 0] = 1.
    trans[1, 1] = 0.

    # assemble the initial condition
    y0 = np.concatenate((C0, N0), axis=0)

    # Create solver object
    vtol = 1.e-8
    solver = MOL(TDR, y0, nop = nop, domain = square, livePlotting=False,
                 noPDEs = size, transitionMatrix = trans,
                 time = time, vtol=vtol)

    solver.run()
    dfs = solver.dfs
    print('keys:',dfs.keys())

    # assume that all the times are the same
    times = dfs[0].columns.values

    time = times[-1]
    yy0 = dfs[0][time].reshape((N,N))
    yy1 = dfs[1][time].reshape((N,N))

    plt.figure(figsize=(15,7))
    plt.subplot(2,2,1, aspect=1)
    plt.title('$C(x, %.4f)$' % time)
    plt.pcolormesh(X, Y, yy0)
    plt.colorbar()

    plt.subplot(2,2,2, aspect=1)
    plt.title('$N(x, %.4f)$' % time)
    plt.pcolormesh(X, Y, yy1)
    plt.colorbar()

    # let's extract a 1d-profile
    yy_exact = nc(time, X, Y)

    plt.subplot(2,2,3, aspect=1)
    plt.title('$C(x, %.4f)$' % time)
    plt.pcolormesh(X, Y, c1(X, Y))
    plt.colorbar()

    plt.subplot(2,2,4, aspect=1)
    plt.title('$N(x, %.4f)$' % time)
    plt.pcolormesh(X, Y, yy_exact)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1, aspect=1)
    plt.title('$C(x, %.4f)$' % time)
    plt.pcolormesh(X, Y, np.abs(c1(X, Y) - yy0))
    plt.colorbar()

    #ax = plt.axes(projection='3d')
    #ax.plot_surface(X, Y, yy0, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    #ax.set_title('surface')

    plt.subplot(1,2,2, aspect=1)
    plt.title('$N(x, %.4f)$' % time)
    plt.pcolormesh(X, Y, np.abs(yy_exact - yy1))
    plt.colorbar()
    plt.show()

    profile = radial_profile(yy1)
    x = np.linspace(0, 0.5, profile.size)
    plt.plot(x, profile, label='Numerical')
    plt.plot(x, n_polar(time, x), label='Analytical')
    plt.xlim([np.min(x), np.max(x)])
    plt.legend(loc='best')
    plt.show()

    x0, y0 = N/2, N/2
    x1, y1 = N-1, N/2
    num = N
    x, y   = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

    # extract values
    zi = scipy.ndimage.map_coordinates(yy1, np.vstack((x,y)))

    fig, axes = plt.subplots(nrows=2)
    #axes[0].imshow(yy1)
    axes[0].pcolormesh(X, Y, yy1)
    x0 /= N
    x1 /= N
    y0 /= N
    y1 /= N
    print(x0, x1, y0, y1)
    axes[0].plot([x0, x1], [y0, y1], 'ro-')
    axes[0].axis('image')

    xs = np.linspace(0, 0.5, N)
    axes[1].plot(xs, zi, label='Numerical')
    axes[1].plot(xs, n_polar(time, xs), label='Analytical')
    plt.legend(loc='best')

    plt.show()


