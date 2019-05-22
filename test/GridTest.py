#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

import unittest
import numpy as np

from tdr.Grid import Grid
from tdr.Domain import Interval
from tdr.helpers import asarray
from utils.utils import array_ptr


class GridTest(unittest.TestCase):

    def test_grid_construction(self):
        # test parameters
        n_eqns = 1
        nop    = 1
        a      = -1
        b      = 1
        bw     = 2

        dom  = Interval(a, b, nop=nop)
        grid = Grid(n_eqns, dom, bw=bw)

        self.assertEqual(len(grid.patches), nop)


    def test_grid_construction_patches(self):
        for nop in np.arange(1, 4, 1):
            # test parameters
            n_eqns = 1
            a      = -1
            b      = 1
            bw     = 2

            dom  = Interval(a, b, nop=nop)
            grid = Grid(n_eqns, dom, bw=bw)

            self.assertEqual(len(grid.patches), nop)

            x0s = dom.x0s

            for i, patch in enumerate(grid.patches):
                self.assertTrue(np.abs(x0s[i] - patch.x0[0]) < 1e-8)

                # also verify by computing xf from patch info
                xf_comp = patch.x0 + patch.shape

                if i < len(grid.patches)-1:
                    self.assertTrue(np.abs(x0s[i+1] - patch.xf[0]) < 1e-8)
                    self.assertTrue(np.abs(x0s[i+1] - xf_comp[0]) < 1e-8)
                else:
                    self.assertTrue(np.abs(b - patch.xf[0]) < 1e-8)
                    self.assertTrue(np.abs(b - xf_comp[0]) < 1e-8)


#if __name__ == '__main__':
#
#    nop = 2
#
#    dom = Interval(-1, 1, nop=nop)
#    print('dom:', dom.bds)
#
#    # test interval first
#    print('dX:', dom.dX, ' shape:', dom.dX.shape, ' dX:', dom.dX[1])
#    print('x0s:', dom.x0s, ' x0:', dom.x0)
#    print('xfs:', dom.xfs, ' xf:', dom.xf)
#    print('N:', dom.N)
#    print('x:', dom.xs())
#
#    ngb = np.asarray(dom.bd).reshape(1)
#    dX  = asarray(dom.dX)
#    x0  = asarray(dom.origin())
#    N   = asarray(dom.N, np.int)
#
#    print('dX:', dX)
#
#    grid = Grid(1, dom, bw=2)
#
#    print('patches:', grid.patches)
#    #print('centers:', grid.cellCentreMatrix)
#
#
#    print('grid:', array_ptr(grid.ydot))
#
#    y = np.arange(0, grid.molsize, 1)
#    y = np.expand_dims(y, axis=0)
#    grid.update(0, y)


if __name__ == '__main__':
    unittest.main()
