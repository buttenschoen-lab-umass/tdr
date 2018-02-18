#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

from Flux import Flux
from utils import apply_along_column

class ReactionFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix):
        super(ReactionFlux, self).__init__(noPDEs, dimensions, transitionMatrix)

        # set priority
        self.priority = 10


    """ i: is the number of PDE """
    def __call__(self, patchData):
        # Compute reaction term for each of the PDEs
        # It seems easier to do this all at once, since these things may depend
        # on each other.
        Hr = apply_along_column(self.trans, patchData.data.y)

        patchData.data.ydot += Hr

        #for i in range(self.n):
        #    self.call(i, patchData)


    def _flux_1d(self, i, patch):
        ri    = self.trans[i]
        p0    = ri(patch.data.y)

        # set ydot in data
        patch.data.ydot[i, :] += p0


if __name__ == '__main__':
    from testing import create_test
    import numpy as np
    ctx = create_test()

    trans = np.eye(2)
    flux = ReactionFlux(ctx, trans)

    for i in range(0, 2):
        flux(i, 0)



