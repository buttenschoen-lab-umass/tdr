#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

from tdr.Flux import Flux
from tdr.utils import apply_along_column


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
        self.call = self._flux_1d


    """ 1D - Implementation """
    def _flux_1d(self, patch):
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



