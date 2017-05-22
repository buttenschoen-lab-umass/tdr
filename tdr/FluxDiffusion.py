#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

from Flux import Flux


class DiffusionFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix):
        super(DiffusionFlux, self).__init__(noPDEs, dimensions, transitionMatrix)

        # set priority
        self.priority = 10


    """ i: is the number of PDE """
    def __call__(self, patchData):
        for i in range(self.n):
            self.call(i, patchData)


    def _flux_1d(self, i, patch):
        pii   = self.trans[i, i]
        uDx   = patch.data.uDx[i, :]
        ydiff = (pii / patch.step_size()) * (uDx[1:] - uDx[:-1])

        # set ydot in data
        patch.data.ydot[i, :] += ydiff


if __name__ == '__main__':
    from testing import create_test
    import numpy as np
    ctx = create_test()

    trans = np.eye(2)
    flux = DiffusionFlux(ctx, trans)

    for i in range(0, 2):
        flux(i, 0)



