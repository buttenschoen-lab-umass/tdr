#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

from tdr.Flux import Flux
from tdr.helpers import apply_along_column_var, apply_along_column
from tdr.helpers import get_function_signatures


class ReactionFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix, *args, **kwargs):
        super(ReactionFlux, self).__init__(noPDEs, dimensions, transitionMatrix, *args, **kwargs)

        # set priority
        self._priority = 5


    """ Name """
    def __str__(self):
        return 'Reaction'


    """ setup function """
    def _setup(self):
        assert self.dim <= 2, 'Reaction Flux not implemented for dimension %d.' % self.dim

        if self.dim == 2:
            self.call = self._flux_2d_const
            return

        # get signature of the functions in self.trans
        sigs = get_function_signatures(self.trans)

        # check if these are unique
        sigs = list({v['pos']:v for v in sigs}.values())
        assert len(sigs)==1, 'We cannot handle with different calling signature for reaction terms yet!'
        sigs = sigs[0]

        # In this case all the density arrays are passed as *args
        if sigs['var']==1:
            if sigs['pos'] == 0:
                self.call = self._flux_1d_const
            elif sigs['pos'] == 1:
                assert False, 'This case is not support yet!'
                #self.rcall = self._flux_1d_const
            elif sigs['pos'] == 2:
                self.call = self._flux_1d_var21
            else:
                assert False, 'ReactionFlux: Should not get here!'
        elif sigs['var'] == 0:
            if sigs['pos'] == self.trans.size:
                self.call = self._flux_1d_const
            elif sigs['pos'] == self.trans.size + 2:
                self.call = self._flux_1d_var21
            else:
                assert False, 'ReactionFlux: Should not get here!'
        else:
            assert False, 'ReactionFlux: Should not get here!'


    """ 1D - Implementation """
    def _flux_1d_var21(self, patch, t):
        # Compute reaction term for each of the PDEs
        # It seems easier to do this all at once, since these things may depend
        # on each other.
        Hr = apply_along_column_var(self.trans, patch.data.y[:, patch.bw:-patch.bw], t, patch.xc)

        # cut of the boundary width
        patch.data.ydot += Hr


    """ 1D - Implementation - const with respect to space and time """
    def _flux_1d_const(self, patch, t):
        # Compute reaction term for each of the PDEs
        # It seems easier to do this all at once, since these things may depend
        # on each other.
        Hr = apply_along_column(self.trans, patch.data.y[:, patch.bw:-patch.bw])

        # cut of the boundary width
        patch.data.ydot += Hr


    """ 2D - Implementation - const with respect to space and time """
    def _flux_2d_const(self, patch, t):
        # Compute reaction term for each of the PDEs
        # It seems easier to do this all at once, since these things may depend
        # on each other.
        Hr = apply_along_column(self.trans, patch.data.y[:, patch.bw:-patch.bw, patch.bw:-patch.bw])

        # cut of the boundary width
        patch.data.ydot += Hr


