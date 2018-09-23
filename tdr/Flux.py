#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

class Flux(object):
    def __init__(self, noPDEs, dimensions, transitionMatrix, *args, **kwargs):
        self.n          = noPDEs
        self.dim        = dimensions
        self.trans      = transitionMatrix
        self.call       = None
        self.bd         = kwargs.pop('boundary', None)

        # priority
        self._priority  = 0

        # possible deformation
        self.cFunctional = kwargs.pop('cFunctional', None)

        # setup
        self._setup()

        # this is a functor so call cannot be None after init
        assert self.call is not None, 'Functor not initialized!'


    """ Calling priority if several fluxes are defined """
    def priority(self):
        return self._priority


    """ Functor """
    def __call__(self, patch):
        self.call(patch)


    """ update constants """
    def update(self, t):
        pass

