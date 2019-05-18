#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

import numpy as np

from tdr.Grid import Grid
from tdr.Domain import Interval
from tdr.helpers import asarray

from utils.utils import array_ptr


if __name__ == '__main__':

    nop = 2

    dom = Interval(-1, 1, nop=nop)
    print('dom:', dom.bds)

    # test interval first
    print('dX:', dom.dX, ' shape:', dom.dX.shape, ' dX:', dom.dX[1])
    print('x0s:', dom.x0s, ' x0:', dom.x0)
    print('xfs:', dom.xfs, ' xf:', dom.xf)
    print('N:', dom.N)
    print('x:', dom.xs())

    ngb = np.asarray(dom.bd).reshape(1)
    dX  = asarray(dom.dX)
    x0  = asarray(dom.origin())
    N   = asarray(dom.N, np.int)

    print('dX:', dX)

    grid = Grid(1, dom, bw=2)

    print('patches:', grid.patches)
    #print('centers:', grid.cellCentreMatrix)


    print('grid:', array_ptr(grid.ydot))

    y = np.arange(0, grid.molsize, 1)
    y = np.expand_dims(y, axis=0)
    grid.update(0, y)


