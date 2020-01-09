#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
from vis.plot_utils import make_segments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vis.utils import clear_figure


def computeTelegrapher(dfs):
    r_dfs = None
    for key, df in dfs.items():
        if r_dfs is None:
            r_dfs = df.copy()
        else:
            r_dfs.add(df, fill_value=0)

    return r_dfs


def getExtremumValues(dfs, vmin=1e-3):
    maxValues = []
    minValues = []

    if isinstance(dfs, dict):
        for key, df in dfs.items():
            maxValue = np.max(df.max().values)
            minValue = max(np.min(df.min().values), vmin)

            maxValues.append(maxValue)
            minValues.append(minValue)
    else:
        maxValues = np.max(dfs.max().values)
        minValues = max(np.min(dfs.min().values), vmin)

    return np.min(minValues), np.max(maxValues)


""" Some plotting helpers """
def plot_x_ticks(axarr):
    # TODO generalize this!
    xlims = axarr.get_xlim()
    L2  = (xlims[0] + xlims[1]) / 2.
    L4  = (xlims[0] + xlims[1]) / 4.
    L34 = 3. * (xlims[0] + xlims[1]) / 4.
    Lp  = xlims[1] + xlims[0]
    axarr.set_xticks((0, L4, L2, L34,  Lp))
    axarr.set_xticklabels(('$0$', '$L/4$', '$L/2$', '$3L/4$', '$L$'))


""" kymograph helpers """
def _kymo_norm_selector(log, minVal, maxVal, eps=1e-3):
    if log:
        # make sure we dont have a zero minimum value in the case of logs
        minVal = max(eps, minVal)
        return LogNorm(vmin=minVal, vmax=maxVal)
    else:
        return Normalize(vmin=minVal, vmax=maxVal)


def _kymo_get_norm(log, dfs, verbose=False, ymin=None, ymax=None, *args, **kwargs):
    minVal, maxVal = ymin, ymax
    if not (minVal is not None and maxVal is not None):
        minVal, maxVal = getExtremumValues(dfs)

    if ymin is not None:
        minVal = ymin

    if ymax is not None:
        maxVal = ymax

    if verbose:
        print('min:', minVal, ' max:', maxVal)
    return _kymo_norm_selector(log, minVal, maxVal)


""" TODO: make local to this file only!!! """
def kymo_get_joint_norm(dfs, log=False, eps=1e-3):
    minVal = 0
    maxVal = 0
    for df in dfs.values():
        minV, maxV = getExtremumValues(df)
        minVal = min(minV, minVal)
        maxVal = max(maxV, maxVal)

    norm = _kymo_norm_selector(log, minVal, maxVal)
    sm   = _kymo_get_mappable(norm)
    return norm, sm


def kymo_get_norms(dfs, log=False, eps=1e-3, *args, **kwargs):
    norms = {}
    sms   = {}
    for key, df in dfs.items():
        norms[key] = _kymo_get_norm(log, df, *args, **kwargs)
        sms[key]   = _kymo_get_mappable(norms[key])

    return norms, sms


def _kymo_get_mappable(norm, cmap=cm.viridis):
    sm   = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm._A = []
    return sm


def _kymo_get_time_max(times, log_y=False):
    if log_y:
        # if we are creating a log plot for time make sure we don't have a
        # minimum that is equal zero.
        tmin, tmax = np.min(times[np.nonzero(times)]), np.max(times)
    else:
        tmin, tmax = np.min(times), np.max(times)

    return tmin, tmax


def _kymo_create_pos(dfs, position):
    pos   = {}

    for j, (col, df) in enumerate(dfs.items()):
        times = df.columns.values

        positions = np.empty((len(times), 2))
        positions[:, 0] = position[0] * np.ones_like(times)
        positions[:, 1] = position[1] * np.ones_like(times)
        pos[j]   = positions

    return pos


def _kymo_create_lines(dfs, position, norm, log_y):
    lines = {}
    pos   = {}
    cols  = len(dfs)

    # make sure we can index it
    if not (isinstance(norm, np.ndarray) or isinstance(norm, list)):
        norms = np.asarray([norm])
    else:
        norms = np.asarray(norm)

    for j, (col, df) in enumerate(dfs.items()):
        times = df.columns.values

        positions = np.empty((len(times), 2))
        positions[:, 0] = position[0] * np.ones_like(times)
        positions[:, 1] = position[1] * np.ones_like(times)
        pos[j]   = positions
        lines[j] = []

        for i, time in enumerate(df.columns.values):
            values   = df[df.columns[i]]
            x_values = np.linspace(position[0], position[1], len(values))
            segments = make_segments(x_values, time * np.ones_like(values))

            # TODO improve this!
            if log_y:
                lw = 4. * (1. + 10. * np.exp(-time))
            else:
                lw = 6.

            lc       = LineCollection(segments, cmap=cm.viridis, norm=norms[j],
                                      lw=lw, zorder=0)
            lc.set_array(values)
            lines[j].append(lc)

    return lines, pos, cols


""" core kymograph plotting """
def plot_kymograph_core(dfs, norm, names=None, position=np.array([0,1]), log=True,
                        log_y=False, axarr=None, dpi=500, xlabels=None,
                        keys=None):
    #norm = _kymo_get_norm(log, dfs)
    #sm   = _kymo_get_mappable(norm)
    times = {}
    tmin, tmax = {}, {}
    for col, df in dfs.items():
        times[col] = df.columns.values
        tmin[col], tmax[col] = _kymo_get_time_max(times[col], log_y=log_y)

    #times = dfs[0].columns.values
    #tmin, tmax = _kymo_get_time_max(times, log_y=log_y)

    lines, pos, cols = _kymo_create_lines(dfs, position, norm, log_y)

    if cols == 1 or (keys is not None and len(keys) == 1):
        axarr = np.asarray(axarr).reshape((1,))

    if keys is None:
        keys = dfs.keys()

    for i, col in enumerate(keys):
        if names is not None:
            axarr[i].set_title(names[i])

        axarr[i].plot(pos[i][:, 0], times[i], color='k')
        axarr[i].plot(pos[i][:, 1], times[i], color='k')

        if xlabels is not None:
            axarr[i].set_xlabel('Domain')

        if log_y:
            axarr[i].set_yscale('log')

        for line in lines[col]:
            axarr[i].add_collection(line)

        # compute x-limits
        xmin = np.min(pos[i][:, 0])
        xmax = np.max(pos[i][:, 1])

        # set xlimits
        axarr[i].set_xlim([xmin, xmax])
        axarr[i].set_ylim([tmin[i], tmax[i]])

    #plt.colorbar(sm)


""" core kymograph plotting """
def plot_kymograph_core_new(dfs, norm, names=None, position=np.array([0,1]),
                            log=True, log_y=False, axarr=None, dpi=500,
                            xlabels=None, keys=None, *args, **kwargs):
    #norm = _kymo_get_norm(log, dfs)
    #sm   = _kymo_get_mappable(norm)
    if isinstance(dfs, pd.DataFrame):
        dfs = {0 : dfs}

    times = {}
    tmin, tmax = {}, {}
    for col, df in dfs.items():
        times[col] = df.columns.values
        tmin[col], tmax[col] = _kymo_get_time_max(times[col], log_y=log_y)

    pos = _kymo_create_pos(dfs, position)

    if keys is None:
        keys = dfs.keys()

    if keys is not None and len(keys) == 1:
        axarr = np.asarray(axarr).reshape((1,))

    for i, col in enumerate(keys):
        if names is not None:
            axarr[i].set_title(names[i])

        # otherwise it's the wrong way around
        data = dfs[col].transpose()
        position = pos[col]
        time = times[col]

        X = np.zeros(data.shape)
        Y = np.zeros(data.shape)

        for k in range(X.shape[0]):
            Y[k, :] = time[k]
            X[k, :] = np.linspace(position[k, 0], position[k, 1], X.shape[1], endpoint=True)

        axarr[i].pcolormesh(X, Y, dfs[col].transpose(), norm=norm)

        if xlabels is not None:
            axarr[i].set_xlabel('Domain')

        if log_y:
            axarr[i].set_yscale('log')

        # compute x-limits
        xmin = np.min(pos[i][:, 0])
        xmax = np.max(pos[i][:, 1])

        # set xlimits
        axarr[i].set_xlim([xmin, xmax])
        axarr[i].set_ylim([tmin[i], tmax[i]])

    #plt.colorbar(sm)


""" TODO: This function is too slow! """
def plot_kymograph(dfs, names, position=np.array([0,1]), log=True, log_y=False,
                   outputname=None, dpi=500, verbose=False, colorbar=True,
                   ymin=None, ymax=None):

    norms = []
    for i, df in enumerate(dfs.values()):
        norm = _kymo_get_norm(log, df, verbose=verbose, ymin=ymin, ymax=ymax)
        norms.append(norm)

    cols  = len(dfs)
    f, axarr = plt.subplots(1, cols, sharey=True, figsize=(10, 5))

    plot_kymograph_core(dfs, norms, names=names, position=position, log=log,
                        log_y=log_y, axarr=axarr)

    axarr[0].set_ylabel('Time')

    # move to core!
    if colorbar:
        sm = _kymo_get_mappable(norm, cmap=cm.viridis)
        plt.colorbar(sm)

    plt.tight_layout()

    if outputname is None:
        plt.show()
    else:
        f.savefig(outputname, dpi=dpi)

        # make sure we don't blow up in memory usage
        clear_figure(f)
