Solver for Taxis-Diffusion-Reaction (TDR) systems
===================================

### TDR (Taxis-Diffusion-Reaction) 
The folder tdr contains the implementation of a finite volume discretization for PDEs. The TDR class provides a right hand side of an ODE system which can be passed to an integrator.


**Supported Terms:**
1. Reaction;  g(t, x, y)
2. Diffusion; D(t)
3. Taxis
4. Non-local adhesion
5. Advection
6. Dilution / Concentration

For more detailed documentation refer to the header of MOL.py, TDR.py, Data.py.


**Supported Boundary conditions:**
1. Neumann (only homogenous)
2. NoFlux  (only homogenous)
3. Dirichlet (not well-tested)
4. NoBc (not tested)

