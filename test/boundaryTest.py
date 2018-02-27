import numpy as np

from mol.MOL import MOL
from tdr.TDR import TDR
from tdr.Boundary import Neumann, DomainBoundary
from visualization.plot_utils import create_plots
#from storage.io import writeDataFrame

from python.utils import Average


if __name__ == '__main__':
    n = 10
    cellsPerUnitLength = 2**n
    h = 1. / (2 ** n)
    L = 2. + 2. * h
    N = L * cellsPerUnitLength
    x = np.linspace(0, L - h, N)
    h = x[1] - x[0]
    y0 = 1. + np.random.normal(loc=0.,scale=.25,size=x.size)
    y0 /= Average(x, y0)
    nop = 1

    lB = Neumann(value = 0, oNormal = -1)
    rB = Neumann(value = 0, oNormal =  1)
    B  = DomainBoundary(lB, rB)

    #ngb = [{ 'left' : 1, 'right' : 1 }]
    ngb = B
    dX  = np.array([[h]])
    x0  = np.array([[0]])
    N   = np.array([[L * cellsPerUnitLength]]).astype(int)

    # number of equations
    size = 1

    # transition matrices
    trans    = np.ones(1).reshape((1,1))
    alpha    = 7.
    M        = Average(x, y0)
    Adhtrans = alpha * np.zeros(1).reshape((1,1))

    # times
    t0 = 0.
    tf = 2.5
    dt = 0.1

    # Create solver object
    solver = MOL(TDR, y0, nop = nop, ngb = ngb, dX = dX, x0 = x0, N = N,
                 noPDEs = size, transitionMatrix = trans, t0 = t0, tf = tf,
                 dt = dt, vtol=1e-3, name='boundaryTest')

    print('Running solver')
    solver.run()

    df = solver.df
    #writeDataFrame('results/boundaryTest.h5', df)

    create_plots(df, x, lw=1.75, ylims=[0.,2.])

