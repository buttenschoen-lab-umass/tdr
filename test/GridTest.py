#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

import unittest
import numpy as np

from tdr.Grid import Grid
from tdr.Domain import Interval
from tdr.Domain import Square
from utils.utils import array_ptr


class GridTest(unittest.TestCase):

    def test_grid_construction_interval(self):
        # test parameters
        n_eqns = 1
        nop    = 1
        a      = -1
        b      = 1
        bw     = 2

        dom  = Interval(a, b, nop=nop)
        grid = Grid(n_eqns, dom, bw=bw)

        self.assertEqual(len(grid.patches), nop)

        for i, patch in enumerate(grid.patches):
            self.assertEqual(patch.dim, 1)


    def test_grid_construction_square(self):
        # test parameters
        n_eqns = 1
        nop    = 1
        a      = -1
        b      = 1
        c      = 1
        d      = 2
        bw     = 2

        dom  = Square(a, b, c, d, nop=nop)
        grid = Grid(n_eqns, dom, bw=bw)

        self.assertEqual(len(grid.patches), nop)

        for i, patch in enumerate(grid.patches):
            self.assertEqual(patch.dim, 2)
            self.assertEqual(patch.N.size, 2)
            self.assertEqual(patch.dX.size, 2)


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
            self.assertLessEqual(np.int(b-a) * 2**6, grid.gridsize)
            self.assertEqual(n_eqns * grid.gridsize, grid.molsize)

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


    def test_grid_construction_patches_many_eqns(self):
        for nop in np.arange(1, 4, 1):
            # test parameters
            n_eqns = 2
            a      = -1
            b      = 1
            bw     = 2

            dom  = Interval(a, b, nop=nop)
            grid = Grid(n_eqns, dom, bw=bw)

            self.assertEqual(len(grid.patches), nop)
            self.assertLessEqual(np.int(b-a) * 2**6, grid.gridsize)
            self.assertEqual(n_eqns * grid.gridsize, grid.molsize)


    def test_grid_update(self):
        # test parameters
        n_eqns = 1
        a      = -1
        b      = 1
        bw     = 2

        dom  = Interval(a, b, nop=1)
        grid = Grid(n_eqns, dom, bw=bw)

        # get ydot pointer
        y_ptr = array_ptr(grid.ydot)

        # create some fake data for the update
        y = np.arange(0, grid.molsize, 1)
        y = np.expand_dims(y, axis=0)
        grid.update(0, y)

        # check that y_ptr is as before
        self.assertEqual(y_ptr, array_ptr(grid.ydot))


    def test_grid_update_patches(self):
        # test parameters
        n_eqns = 1
        a      = -1
        b      = 1
        bw     = 2

        dom  = Interval(a, b, nop=2)
        grid = Grid(n_eqns, dom, bw=bw)

        # get ydot pointer
        y_ptr = array_ptr(grid.ydot)

        # create some fake data for the update
        y = np.arange(0, grid.molsize, 1)
        y = y.reshape((1, grid.gridsize))
        grid.update(0, y)

        # check that y_ptr is as before
        self.assertEqual(y_ptr, array_ptr(grid.ydot))

        # get patches from grid
        patch1 = grid.patches[0]
        patch2 = grid.patches[1]

        # shortcuts
        y1 = patch1.y
        y2 = patch2.y

        # check patch1 left boundary
        self.assertTrue(np.all(y1[:, 0:bw]==np.flip(y1[:, bw:bw+bw])))

        # check glue between patch1 and patch2
        self.assertTrue(np.all(y1[:, -bw:]==y2[:, bw:bw+bw]))
        self.assertTrue(np.all(y1[:, -bw-bw:-bw]==y2[:, 0:bw]))

        # check patch2 right boundary
        self.assertTrue(np.all(y2[:, -bw-bw:-bw]==np.flip(y2[:, -bw:])))


    def test_grid_update_many_eqs_patches(self):
        # test parameters
        n_eqns = 2
        a      = -1
        b      = 1
        bw     = 2

        dom  = Interval(a, b, nop=2)
        grid = Grid(n_eqns, dom, bw=bw)

        # get ydot pointer
        y_ptr = array_ptr(grid.ydot)

        # create some fake data for the update
        y = np.arange(0, grid.molsize, 1)
        y = y.reshape((2, grid.gridsize))
        grid.update(0, y)

        # check that y_ptr is as before
        self.assertEqual(y_ptr, array_ptr(grid.ydot))

        # get patches from grid
        patch1 = grid.patches[0]
        patch2 = grid.patches[1]

        # shortcuts
        y1 = patch1.y
        y2 = patch2.y

        # check patch1 left boundary
        self.assertTrue(np.all(y1[:, 0:bw]==np.flip(y1[:, bw:bw+bw], axis=1)))

        # check glue between patch1 and patch2
        self.assertTrue(np.all(y1[:, -bw:]==y2[:, bw:bw+bw]))
        self.assertTrue(np.all(y1[:, -bw-bw:-bw]==y2[:, 0:bw]))

        # check patch2 right boundary
        self.assertTrue(np.all(y2[:, -bw-bw:-bw]==np.flip(y2[:, -bw:], axis=1)))


    def test_grid_update_many_eqs_patches2(self):
        # test parameters
        n_eqns = 2
        a      = -1
        b      = 1
        bw     = 2

        dom  = Interval(a, b, nop=3)
        grid = Grid(n_eqns, dom, bw=bw)

        # get ydot pointer
        y_ptr = array_ptr(grid.ydot)

        # create some fake data for the update
        y = np.arange(0, grid.molsize, 1)
        y = y.reshape((2, grid.gridsize))
        grid.update(0, y)

        # check that y_ptr is as before
        self.assertEqual(y_ptr, array_ptr(grid.ydot))

        # get patches from grid
        patch1 = grid.patches[0]
        patch2 = grid.patches[1]
        patch3 = grid.patches[2]

        # shortcuts
        y1 = patch1.y
        y2 = patch2.y
        y3 = patch3.y

        # check patch1 left boundary
        self.assertTrue(np.all(y1[:, 0:bw]==np.flip(y1[:, bw:bw+bw], axis=1)))

        # check glue between patch1 and patch2
        self.assertTrue(np.all(y1[:, -bw:]==y2[:, bw:bw+bw]))
        self.assertTrue(np.all(y1[:, -bw-bw:-bw]==y2[:, 0:bw]))

        # check glue between patch2 and patch3
        self.assertTrue(np.all(y2[:, -bw:]==y3[:, bw:bw+bw]))
        self.assertTrue(np.all(y2[:, -bw-bw:-bw]==y3[:, 0:bw]))

        # check patch3 right boundary
        self.assertTrue(np.all(y3[:, -bw-bw:-bw]==np.flip(y3[:, -bw:], axis=1)))


    # make sure ydot is set correctly when using patches
    def test_grid_ydot_set(self):
        # test parameters
        n_eqns = 2
        a      = -1
        b      = 1
        bw     = 2

        dom  = Interval(a, b, nop=2)
        grid = Grid(n_eqns, dom, bw=bw)

        # create some fake data for the update
        y = np.arange(0, grid.molsize, 1)
        y = y.reshape((2, grid.gridsize))

        # set ydot of grid to something non-zero for the test
        grid.ydot.fill(1)

        # first test grid
        grid.update(0, y)
        self.assertTrue(np.all(grid.dy == 0.))

        # now do it patch by patch
        patch1 = grid.patches[0]
        patch2 = grid.patches[1]

        # reset ydot
        grid.ydot.fill(1)

        # update patch1
        patch1.update(0, y)
        self.assertTrue(np.all(grid.dy[:, patch1.ps:patch1.pe] == 0.))
        self.assertTrue(np.all(grid.dy[:, patch2.ps:patch2.pe] == 1.))

        # reset ydot
        grid.ydot.fill(1)

        # update patch2
        patch2.update(0, y)
        self.assertTrue(np.all(grid.dy[:, patch1.ps:patch1.pe] == 1.))
        self.assertTrue(np.all(grid.dy[:, patch2.ps:patch2.pe] == 0.))


if __name__ == '__main__':
    unittest.main()
