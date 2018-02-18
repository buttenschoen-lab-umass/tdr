#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

class Flux(object):
    def __init__(self, noPDEs, dimensions, transitionMatrix):
        self.n        = noPDEs
        self.dim      = dimensions
        self.trans    = transitionMatrix
        self.call     = None

        # priority
        self.priority = 0

        # setup
        self._setup()


    def priority(self):
        return self.priority


    def _setup(self):
        if self.dim == 1:
            pass
        else:
            assert False, 'At the moment we only support 1D simulations.'


    def __call__(self, patch):
        pass


    """ stubs """
    def _flux_1d_periodic(self, i, patch):
        assert False, 'Periodic flux in 1D is not implemented!'


    def _flux_1d_neumann(self, i, patch):
        assert False, 'Neumann flux in 1D is not implemented!'


    def _flux_1d_dirichlet(self, i, patch):
        assert False, 'Dirichlet flux in 1D is not implemented!'

