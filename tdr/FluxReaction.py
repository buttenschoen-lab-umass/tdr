#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

from tdr.Flux import Flux
from tdr.utils import apply_along_column_var, apply_along_column

from inspect import signature, Parameter


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
        assert self.dim == 1, 'Reaction Flux not implemented for dimension %d.' % self.dim

        # get signature of the functions in self.trans
        sigs = []
        for i, func in enumerate(self.trans):
            pars = signature(func).parameters

            ptype = {'var' : 0, 'pos' : 0}
            # iterate over pars
            for par in pars.values():
                if par.kind == Parameter.VAR_POSITIONAL:
                    ptype['var'] += 1
                elif par.kind == Parameter.POSITIONAL_OR_KEYWORD or par.kind == Parameter.POSITIONAL_ONLY:
                    ptype['pos'] += 1
                else:
                    assert False, 'Dealing with unknown parameter kind %s.' % par.kind

            sigs.append(ptype)

        # check if these are unique
        sigs = list({v['pos']:v for v in sigs}.values())
        assert len(sigs)==1, 'We cannot handle with different calling signature for reaction terms yet!'
        sigs = sigs[0]
        assert sigs['var']==1, 'Variable positional arguments must be one!'

        if sigs['pos'] == 0:
            self.call = self._flux_1d_const
        elif sigs['pos'] == 1:
            assert False, 'This case is not support yet!'
            #self.rcall = self._flux_1d_const
        elif sigs['pos'] == 2:
            self.call = self._flux_1d_var21
        else:
            assert False, 'ReactionFlux: Should not get here!'


    """ 1D - Implementation """
    def _flux_1d_var21(self, patch, t):
        # Compute reaction term for each of the PDEs
        # It seems easier to do this all at once, since these things may depend
        # on each other.
        Hr = apply_along_column_var(self.trans, patch.data.y[:, patch.bW:-patch.bW], t, patch.xc)

        # cut of the boundary width
        patch.data.ydot += Hr


    """ 1D - Implementation - const with respect to space and time """
    def _flux_1d_const(self, patch, t):
        # Compute reaction term for each of the PDEs
        # It seems easier to do this all at once, since these things may depend
        # on each other.
        Hr = apply_along_column(self.trans, patch.data.y[:, patch.bW:-patch.bW])

        # cut of the boundary width
        patch.data.ydot += Hr


if __name__ == '__main__':
    from testing import create_test
    import numpy as np
    ctx = create_test()

    trans = np.eye(2)
    flux = ReactionFlux(ctx, trans)

    for i in range(0, 2):
        flux(i, 0)



