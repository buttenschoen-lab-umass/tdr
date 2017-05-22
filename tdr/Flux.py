#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

class Flux(object):
    def __init__(self, noPDEs, dimensions, transitionMatrix):
        self.n      = noPDEs
        self.dim    = dimensions
        self.trans  = transitionMatrix
        self.call   = None

        # priority
        self.priority = 0

        # setup
        self._setup()


    def priority(self):
        return self.priority


    def _setup(self):
        if self.dim == 1:
            self.call = self._flux_1d
        else:
            assert False, ''


    def __call__(self, patch):
        pass


