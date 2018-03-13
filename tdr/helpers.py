#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np


def getHillFunction(n):
    hillFunction = lambda x : np.power(x, n) / (1. + np.power(x, n))
    return hillFunction


def getStepFunction(x, step_point = 0.1, hfactor = 2.):
    return np.piecewise(x, [x < step_point, x >= step_point], [hfactor, .1])


