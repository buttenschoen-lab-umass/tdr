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



# Tutorial

**This is a work in progress; not tested yet. I will add these as a example file**

1. How to setup a TDR simulation. For this example we will use the scalar
   non-local adhesion problem.

We start by creating a domain.
```

    bc = DomainBoundary(left=Periodic(), right=Periodic())

    # The interval will have 2**n divisions per unit length.
    n = 10
    interval = Interval(0, 1, n=n, bd=bd)
```
Next we setup the transition matrices, the diagonal elements of trans define
the diffusion coefficients, while the off-diagonal elements define the taxis
strength coefficients. That means that the entry (i, i) is the diffusion
coefficient of population i; while entry (i, j) is the taxis coefficient of
population i with respect to gradients of population j. The elements of AdhTrans 
define the adhesion coefficients, and work the same as the elements of trans.

```
    # transition matrices
    trans    = np.ones(1).reshape((1,1))
    Adhtrans = alpha * np.ones(1).reshape((1,1))
```
Next we setup the time control for the simulation, and output.
```
    # setup simulation time
    no_data_points = 100

    dt = max((tf - t0) / no_data_points, 0.01)
    time = Time(t0 = t0, tf = tf, dt = dt)

    # the filename to save simulation data
    name = basename + '_L=%f_alpha=%f_tf=%f' % (L, alpha, tf)
```
If using numpy + rowmap compiled against mkl we can control multi-core support
```
    set_num_threads(threads)
```
Finally we are ready to create the complete solver object.
```
    # Finally setup the core solver.
    solver = MOL(TDR, y0, domain=interval, transitionMatrix=trans, AdhesionTransitionMatrix=Adhtrans,
                 time=time, vtol=vtol, name=name, outdir=outdir, 
                 verbose=verbose, max_iter=100000, *args, **kwargs)
```
The solver is now executed using
```
    solver.run()
```


2. Add *rowmap* to the tdr repository.

```
git submodule add git@github.com:adrs0049/rowmap.git details
```

This is a bit irritating. Maybe there is a better way.

```
for dir in $(ls details/); do [[ -d details/$dir ]] && ln -s details/$dir $dir ; done
```




