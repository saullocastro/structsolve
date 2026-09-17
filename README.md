Github Actions status:
[![pytest](https://github.com/saullocastro/structsolve/actions/workflows/pytest.yml/badge.svg)](https://github.com/saullocastro/structsolve/actions/workflows/pytest.yml)
[![Deploy](https://github.com/saullocastro/structsolve/actions/workflows/pythonpublish.yml/badge.svg)](https://github.com/saullocastro/structsolve/actions/workflows/pythonpublish.yml)

Coverage status:
[![codecov](https://github.com/saullocastro/structsolve/actions/workflows/coverage.yml/badge.svg)](https://github.com/saullocastro/structsolve/actions/workflows/coverage.yml)
[![Codecov Status](https://codecov.io/gh/saullocastro/structsolve/branch/main/graph/badge.svg?token=NNHK0SFZNH)](https://codecov.io/gh/saullocastro/structsolve)

Structural analysis solvers tailored for semi-analytical models
===============================================================

- Linear statics: [K]{u} = {f}
- Eigensolver for Linear buckling: ([K] + lambda[KG]){u} = 0
- Eigensolver for dynamics: ([K] + lambda^2[M]){u} = 0
- Nonlinear statics using Newton-Raphson 
- Nonlinear statics using the Arc-Length method

Currently these solvers are pretty much compatible with my other repositories
[panels](https://github.com/saullocastro/panels), 
[buckling](https://github.com/saullocastro/buckling).


Citing this library
===================

Saullo G. P. Castro (2026). Structural analysis solvers tailored for semi-analytical models (Version 0.4.3). Zenodo. DOI: https://doi.org/10.5281/zenodo.2581212.


Documentation
=============

The documentation is available on: https://saullocastro.github.io/structsolve.



History
=======

See [CHANGELOG.md](CHANGELOG.md) for the details of each version.

* version 0.4.3 (2026-09-17)
    - Fixed the eigenvalue shift of the sparse solver of `lb`, which could
      return buckling loads higher than the critical ones
    - Fixed wrong buckling loads of the sparse solver of `lb` when SciPy is
      linked against Intel MKL 2024.2.0 to 2025.0.0 (e.g. Anaconda on
      Windows), whose `dsteqr` routine breaks ARPACK
    - `lb` condenses out the dofs where `KG` is null and solves small
      condensed problems with a dense solver
    - `lb` verifies the eigenpairs of the sparse solver, raising an error
      instead of returning wrong results
    - The sparse solver of `lb` returns the load multipliers sorted as the
      dense solver
    - Newton-Raphson with full Newton iterations and a relative convergence
      criterion by default, reaching quadratic convergence with exact tangent
      stiffness matrices
    - Fixed the load-factor bookkeeping of Newton-Raphson, which could finish
      before reaching the full load
    - Arc-length methods (Riks and Crisfield) rewritten, able to trace limit
      points, snap-through and snap-back until a load factor of exactly 1.0
    - Sphinx documentation
* version 0.3.1 (2026-04-09)
    - Robust estimation of the eigenvalue shift for singular matrices in
      `freq` and `lb`, and new `skip_null_cols` argument
    - Python 3.14 support
    - Estimation of the eigenvalue shift in `freq` and `lb`
    - Packaging with `pyproject.toml`, GitHub Actions and a test suite
* version 0.2.2 (2019-03-02)
    - Release of 0.2.1 with an updated distribution
    - Fixed the dense solver of `freq`
    - `freq` returns the eigenvalues `lambda**2` instead of the natural
      frequencies
* version 0.1.0 (2018-06-23)
    - First release, with the solvers from `compmech`

License
-------
Distributed in the 3-Clause BSD license (https://raw.github.com/saullocastro/structsolve/master/LICENSE).

Contact: S.G.P.Castro@tudelft.nl

