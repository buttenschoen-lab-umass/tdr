#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import absolute_import, print_function, division

from collections import OrderedDict
import numpy as np

from iout.io import load_datafile
from vis.kymograph import plot_x_ticks, kymo_get_norms, plot_kymograph_core_new
import matplotlib.pyplot as plt
from matplotlib import animation


def get_yticks(y_max, no_ticks=4, decimals=1):
    y_vals = []
    fraction = 1. / no_ticks
    for i in range(no_ticks+1):
        y_vals.append(round(2 * np.round(i * fraction * y_max, decimals=decimals))/2)
    return y_vals


def plot_profiles_core(dfs, interval, axarr, title_str='$\\alpha = %s$'):
    y_max = 0
    y_maxs = []
    y_mins = []

    for j, (alpha, df) in enumerate(dfs.items()):
        x = interval.xs()
        y = df[df.columns[-1]]
        y_max = max(np.max(y), y_max)
        y_maxs.append(np.max(y))
        y_mins.append(np.min(y))

        if not isinstance(axarr, np.ndarray) and not isinstance(axarr, list):
            axarr = [axarr]

        if title_str is not None:
            axarr[j].set_title(title_str % str(alpha))
        axarr[j].plot(x, y, color='k', lw=2)
        axarr[j].set_xlim([np.min(x), np.max(x)])

        # plot new tick labels
        plot_x_ticks(axarr[j])

    for i, ax in enumerate(axarr):
        y_min = 0.9 * y_mins[i]
        y_max = 1.1 * y_maxs[i]
        diff = y_max - y_min
        if diff < 0.1:
            y_min -= 0.1
            y_max += 0.1

        if y_min < 0.1:
            y_min = 0.

        ax.set_ylim([y_min, y_max])
        #ax.set_yticks(get_yticks(y_maxs[i], 4))


def plot_osc_core(dfs, axarr, title_str=None):
    y_max = 0
    y_maxs = []
    y_mins = []

    for j, (alpha, df) in enumerate(dfs.items()):
        # need to somehow compute the spatial average now
        x = df.columns.values

        print('Found %d time points.' % len(df.columns))
        y_avg = np.empty(len(df.columns))

        for k in range(len(df.columns)):
            y_avg[k] = np.average(df[df.columns[k]])

        y_max = max(np.max(y_avg), y_max)
        y_maxs.append(np.max(y_avg))
        y_mins.append(np.min(y_avg))

        if not isinstance(axarr, np.ndarray) and not isinstance(axarr, list):
            axarr = [axarr]

        if title_str is not None:
            axarr[j].set_title(title_str[j])

        axarr[j].plot(x, y_avg, color='k', lw=2)
        axarr[j].set_xlim([np.min(x), np.max(x)])

        # plot new tick labels
        # plot_x_ticks(axarr[j])

    for i, ax in enumerate(axarr):
        y_min = 0.9 * y_mins[i]
        y_max = 1.1 * y_maxs[i]
        diff = y_max - y_min
        if diff < 0.1:
            y_min -= 0.1
            y_max += 0.1

        if y_min < 0.1:
            y_min = 0.

        ax.set_ylim([y_min, y_max])
        #ax.set_yticks(get_yticks(y_maxs[i], 4))


def plot_profiles(data, intervals, outputname=None, *args, **kwargs):
    nx = 1
    ny = len(data)
    f, axarr = plt.subplots(nx, ny, sharey=False, figsize=(10, 5))

    plot_profiles_core(data, intervals, axarr, *args, **kwargs)

    plt.tight_layout()
    if outputname is not None:
        plt.savefig(outputname)


""" Plot every population contained in a data file """
def plot_osc(datafile, figsize=(10,5), outputname=None, dpi=500,
             log=True, titles=None, ymin=0, ymax=4, xmin=0, xmax=None, *args, **kwargs):
    # load the datafile
    dfs = load_datafile(datafile)
    dfs = OrderedDict(sorted(dfs.items(), key=lambda t : t[0]))

    # let's do some transposing
    for k, df in dfs.items():
        dfs[k] = df.transpose()

    # Determine plot dimensions
    nx = 1
    ny = len(dfs)

    print('Creating a plot with dim (%d, %d).' % (nx, ny))

    f, axarr = plt.subplots(nx, ny, figsize=figsize)
    if axarr.ndim == 1:
        axarr = axarr.reshape((axarr.shape[0], 1))

    # TODO avoid loading data twice!
    plot_osc_core(dfs, axarr[:, 0], title_str=titles)

    # set label
    axarr[0, 0].set_ylabel('Amplitude')
    axarr[0, 0].set_xlabel('Time')
    axarr[1, 0].set_xlabel('Time')

    # set some ylims
    axarr[0, 0].set_ylim([ymin, ymax])
    axarr[1, 0].set_ylim([ymin, ymax])

    # set some xlims
    if xmax is not None:
        axarr[0, 0].set_xlim([xmin, xmax])
        axarr[1, 0].set_xlim([xmin, xmax])

    plt.tight_layout()
    if outputname is not None:
        f.savefig(outputname, dpi=dpi)


""" Plot every population contained in a data file """
def plot_figure(datafile, interval, figsize=(10,5), outputname=None, dpi=500,
                log=True, *args, **kwargs):
    # load the datafile
    dfs = load_datafile(datafile)
    dfs = OrderedDict(sorted(dfs.items(), key=lambda t : t[0]))

    # let's do some transposing
    for k, df in dfs.items():
        dfs[k] = df.transpose()

    # Determine plot dimensions
    nx = 2
    ny = len(dfs)

    print('Creating a plot with dim (%d, %d).' % (nx, ny))

    f, axarr = plt.subplots(nx, ny, figsize=figsize)
    if axarr.ndim == 1:
        axarr = axarr.reshape((axarr.shape[0], 1))

    # TODO avoid loading data twice!
    plot_profiles_core(dfs, interval, axarr[0, :])

    norms, sms = kymo_get_norms(dfs, *args, **kwargs)
    for i, alpha in enumerate(dfs.keys()):
        plot_kymograph_core_new(dfs[alpha], norms[alpha], axarr=axarr[1, i],
                                *args, **kwargs)
        plot_x_ticks(axarr[1, i])

    # set label
    axarr[0, 0].set_ylabel('Amplitude')
    axarr[1, 0].set_ylabel('Time')

    #f.subplots_adjust(right=0.8)
    #cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    for i, alpha in enumerate(dfs.keys()):
        f.colorbar(sms[alpha], ax=axarr[1, i])

    plt.tight_layout()
    if outputname is not None:
        f.savefig(outputname, dpi=dpi)


def plot_reflected(dfs, interval, L):
    df = dfs[0]
    x = interval.x
    y = df[df.columns[-1]]
    print('y:',y.shape,' x:',x.shape)
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='orig')
    plt.plot(x, y[::-1], label='refl')
    plt.legend(loc='best')
    plt.grid()
    plt.xlim([0, L])


def plot_final_soln(dfs, interval, title='$u(x,t)$', outputname=None, dpi=500,
                   ymin=None, ymax=None, *args, **kwargs):
    df = dfs[0]
    y = df[df.columns[-1]]

    plt.title(title)
    plt.plot(interval.xs(), y, *args, **kwargs)
    plt.xlim([interval.x0, interval.xf])

    if ymin is not None and ymax is not None:
        plt.ylim([ymin, ymax])

    plt.xlabel('Domain')
    plt.ylabel('$u(x,t)$')

    if outputname is None:
        plt.show()
    else:
        plt.savefig(outputname, dpi=dpi)


""" Functions to create animations """
def get_max_values(dfs):
    return np.max(dfs.values)


def create_animation(datafile, L, col=0, basename='Unnamed'):
    dfs = load_datafile(datafile)
    for k, df in dfs.items():
        dfs[k] = df.transpose()

    # get sorted dict keys
    keys = sorted(dfs.keys())

    ymax = np.round(1.5 * get_max_values(dfs[keys[col]]), decimals=1)
    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(0, ymax))
    plt.title(basename)
    line, = ax.plot([], [], lw=2)

    title = ax.text(0.90, 0.90, "", bbox={'facecolor' : 'w', 'alpha' : 0.5,
                                         'pad' : 5}, transform=ax.transAxes, ha='center')

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        columns = dfs[keys[col]].columns
        y = dfs[keys[col]][columns[i]]
        x = np.linspace(0, L, y.size)
        line.set_data(x, y)
        title.set_text('t = %.2g' % columns[i])
        return line,

    frames = len(dfs[keys[col]].columns)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=20, blit=True)
    return anim


def create_animation_double(datafile, L, col=0, basename='Unnamed'):
    dfs = load_datafile(datafile)
    for k, df in dfs.items():
        dfs[k] = df.transpose()

    # get sorted dict keys
    keys = sorted(dfs.keys())

    ymax = np.round(1.5 * get_max_values(dfs[keys[col]]), decimals=1)
    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(0, ymax))
    plt.title(basename)

    title = ax.text(0.90, 0.90, "", bbox={'facecolor' : 'w', 'alpha' : 0.5,
                                         'pad' : 5}, transform=ax.transAxes, ha='center')

    lines = []
    for k in range(len(dfs)):
        line, = ax.plot([], [], lw=2)
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([], [])
        return tuple(lines)

    def animate(i):
        for k, line in enumerate(lines):
            columns = dfs[keys[k]].columns
            y = dfs[keys[k]][columns[i]]
            x = np.linspace(0, L, y.size)
            line.set_data(x, y)

        title.set_text('t = %.2g' % columns[i])
        return tuple(lines)

    frames = len(dfs[keys[col]].columns)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=20, blit=True)
    return anim
