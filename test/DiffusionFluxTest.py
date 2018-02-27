
import sys
from testing import create_test

import numpy as np
from tdr.FluxDiffusion import DiffusionFlux
from tdr.Boundary import DomainBoundary, Neumann, Periodic


if __name__ == '__main__':
    print('BEGINTEST')

    lB = Neumann(0, oNormal = -1)
    rB = Neumann(0, oNormal = 1)
    ngb = DomainBoundary(lB, rB)
    grid, grd = create_test(ngb)

    trans = np.eye(2)

    flux = DiffusionFlux(grd['noPDEs'], grd['dim'], trans)

    grid.apply_flux(flux)

    print('Final fluxes for neumann bc')
    for patch in grid:
        print('size:', patch.data.ydot.size,' ', patch.data.ydot[0,:])

    sys.exit()

    lB = Periodic()
    rB = Periodic()
    ngb = DomainBoundary(lB, rB)
    grid, grd = create_test(ngb)

    flux = DiffusionFlux(grd['noPDEs'], grd['dim'], trans)

    grid.apply_flux(flux)

    print('Final fluxes for periodic bc')
    for patch in grid:
        print('size:', patch.data.ydot.size,' ', patch.data.ydot[0,:])




