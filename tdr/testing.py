import numpy as np
from Data import Data
from Grid import Grid
from context import GlobalSolverContext

def create_test():
    # create data object
    concs = 2
    data = Data(concs, 1)

    L = 10
    cellsPerUnitLength = 2**5
    n = L * cellsPerUnitLength
    h = 1. / (2**5)
    x = np.linspace(0, L, n)
    y1 = np.sin(2. * np.pi * x / L)
    y2 = np.cos(2. * np.pi * x / L)

    nop = 1
    ngb = np.array([[1, 1]])
    dX  = np.array([[h]])
    x0  = np.array([[0]])
    N   = np.array([[L * cellsPerUnitLength]]).astype(int)

    grd = { 'nop' : nop, 'ngb' : ngb, 'dX' : dX, 'N' : N, 'x0' : x0}

    grid = Grid(concs, grd, 1)
    grid.boundaryWidth = 2

    data.boundaryWidth = 2
    data.set_values(np.vstack((y1, y2)))
    data.h = h
    data._compute_uDx()
    data._compute_uAvx()

    ctx = GlobalSolverContext()
    ctx.data = data
    ctx.grd  = grid

    return ctx


