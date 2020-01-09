#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt


""" Try to clear all of a figure to avoid memory usage blow-ups """
def clear_figure(fig):
    plt.close(fig)

    # get all axes and clear them
    axes = fig.get_axes()
    for ax in axes:
        ax.clear()

    # clear the figure
    fig.clf()


""" Get extremum values in a dictionary of dataframes """
def get_max_values(dfs):
    maxs = []
    for key, value in dfs.items():
        maxs.append(np.max(value.values))
    return np.asarray(maxs)


def get_min_values(dfs):
    mins = []
    for key, value in dfs.items():
        mins.append(np.min(value.values))
    return np.asarray(mins)


def get_extrema(dfs):
    mins = get_min_values(dfs)
    maxs = get_max_values(dfs)
    return {'min' : mins, 'max' : maxs}



