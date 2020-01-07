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


    """ Initialize temporaries etc. """
    def init(self, patch):
        pass


    """ Helper to attach objects to self """
    def _attach_object(self, otype, oname, *args, **kwargs):
        if not hasattr(self, oname):
            setattr(self, oname, otype(*args, **kwargs))
        # else:
        # we do nothing!


    """ Functor """
    def __call__(self, patch, t):
        self.call(patch, t)


    """ Name """
    def __str__(self):
        return 'Flux'


    """ update time-dependence in transition matrix values """
    def update(self, t):
        pass


    """ EXPERIMENTAL: update trans for bifurcation continuations """
    def update_trans(self, trans):
        self.trans = trans



