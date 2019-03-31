#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

"""
    This class implements a simple dilution effect in concentration fields due
    to domain size changes in growing domains.
"""

from tdr.Flux import Flux


class DilutionFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix, *args, **kwargs):
        super(DilutionFlux, self).__init__(noPDEs, dimensions, transitionMatrix, *args, **kwargs)

        # set priority
        self._priority = 5


    """ Name """
    def __str__(self):
        return 'Dilution'


    """ setup function """
    def _setup(self):
        assert self.dim == 1, 'Dilution Flux not implemented for dimension %d.' % self.dim
        self.call = self._flux_1d


    """ 1D - Implementation """
    def _flux_1d(self, patch, t):
        # Compute reaction term for each of the PDEs
        # It seems easier to do this all at once, since these things may depend
        # on each other.
        #print('\tr_dot_over_r:', patch.data.r_dot_over_r, ' Hd:', \
        #      patch.data.r_dot_over_r * np.max(patch.data.y[:, patch.bW:-patch.bW]),
        #      ' ',
        #      patch.data.r_dot_over_r * np.min(patch.data.y[:, patch.bW:-patch.bW]))

        Hd = patch.data.y[:, patch.bW:-patch.bW] * patch.data.r_dot_over_r

        # cut of the boundary width
        patch.data.ydot -= Hd


