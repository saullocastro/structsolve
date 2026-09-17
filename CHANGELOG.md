# Changelog

## 0.4.3 (2026-09-17)

### Breaking: new defaults of the non-linear solvers

The Newton-Raphson defaults were tuned around models whose tangent stiffness
matrix was not the exact derivative of the internal force vector, and the
convergence check depended on the units of the model. With an exact tangent
matrix, full Newton-Raphson now converges quadratically. The attribute names
of `Analysis` and the signatures of the callables `calc_fext`, `calc_fint`,
`calc_kC` and `calc_kG` did not change, but the following defaults did:

- `Analysis.modified_NR`: `True` -> `False`, i.e. full Newton-Raphson, with the
  tangent stiffness matrix updated at every iteration.
- `Analysis.line_search`: `True` -> `False`. When activated, the line-search
  is now a safeguard: the full Newton step is tried first and only reduced when
  it fails a sufficient-decrease test on the residual norm.
- `Analysis.relTOL`: `1e-3` (unused) -> `1e-6`, now used by all non-linear
  solvers: the convergence is achieved when
  `||R|| <= relTOL*max(||fext||, ||fint||)`.
- `Analysis.absTOL`: `1e-3` -> `None`. When defined, it is an additional
  absolute criterion on `max(|R|)`, which depends on the units of the model.
- `Analysis.maxNumIter` now counts the number of corrections (linear solutions)
  of each step, and a step can converge at its first residual evaluation.
- A failed step is repeated with the load increment multiplied by `0.5` (was
  `0.9`). The increment grows by `1.1111` only after a step that converged
  without being cut.
- `Analysis.kT_initial_state` is only used when `modified_NR=True`.

### Breaking: arc-length solvers rewritten

The solvers `NL_method='arc_length_riks'` and `NL_method='arc_length_crisfield'`
did not converge in practice. They now share one implementation,
`structsolve.arc_length._solver_arc_length`:

- The arc length is dimensionless, with displacements scaled by the linear
  solution for a load factor of 1, so the analysis no longer depends on the
  units of the model. `initialInc` is the load factor increment of the first
  step along the initial tangent, `maxInc` and `minInc` are the maximum and
  minimum arc-length increments, and `maxArcLength` is the maximum cumulative
  arc length.
- Tangent predictor whose direction follows the previous increment, allowing
  limit points, snap-through and snap-back to be traced.
- Riks: the arc-length constraint is linearized at every iteration (updated
  normal plane), giving quadratic convergence. Crisfield: the quadratic
  constraint is solved exactly, choosing the root with the smallest angle to
  the current increment.
- The bordered system is solved with two solutions of one sparse LU
  factorization, instead of a dense matrix.
- The arc length adapts to the number of iterations of each step.
- **The analysis stops when the load factor reaches exactly 1.0**, besides the
  limits given by `maxArcLength` and `minInc`. Previously Riks ran until
  `maxArcLength`, and Crisfield until 1000 steps or `minInc`.
- `modified_NR` and `compute_every_n` control how often the tangent stiffness
  matrix is updated, as in the Newton-Raphson solver, so the new default
  `modified_NR=False` also applies here.

### Linear buckling: sparse solver of `lb`

The sparse solver of `lb` could return wrong load multipliers without any
warning, for two independent reasons, both fixed:

- **Wrong shift.** The shift `sigma` of the Cayley mode was a single Rayleigh
  quotient, in which positive and negative eigenvalues cancel, and could land
  below the largest `|mu|` of `KG u = mu K u`. The eigenvalue solver then
  returned the eigenvalues nearest to the shift, i.e. load multipliers higher
  than the critical ones, e.g. for a cylinder model from panels. The shift is
  now estimated with power iterations on `K^-1 KG`, giving the largest `|mu|`,
  multiplied by a safety factor of 10. When `K` is singular, not positive
  definite, or the linear solution is inaccurate, the shift falls back to `1.`.
- **Intel MKL bug.** When SciPy is linked against Intel MKL 2024.2.0 to
  2025.0.0, e.g. the current Anaconda builds of SciPy on Windows, ARPACK
  returned wrong load multipliers at random, some of them far below the
  critical load, or raised `ArpackError -8`. The `dsteqr` routine of these MKL
  versions returns wrong eigenvectors for matrices larger than 32 x 32, which
  ARPACK uses for the Ritz vectors when `ncv > 32`, e.g. `num_eigvalues=25`
  gives `ncv=51`. SciPy itself is not affected: the PyPI wheels of SciPy 1.16
  and 1.17 give correct results, and so does the same Anaconda SciPy binary
  with MKL 2025.0.1 or newer.

The sparse solver was changed accordingly:

- The dofs where `KG` has null rows, typically the in-plane dofs, are condensed
  out with a sparse factorization of `K`. When at most `max_dense_size` (new
  argument, default `2000`) dofs remain, the condensed problem is solved with
  the dense `scipy.linalg.eigh`, otherwise with `eigsh` in Cayley mode.
- The eigenpairs are verified: the relative residual
  `||K u + lambda KG u|| / (||K u|| + |lambda| ||KG u||)` must not exceed
  `check_rtol` (new argument, default `1e-3`), and, when `K` is positive
  definite, `K + s KG` must also be positive definite for `s` slightly below
  the lowest positive load multiplier found, i.e. no lower load multiplier
  was missed. When the condensed solution fails, `eigsh` is tried, and a
  `RuntimeError` is raised when no solution passes the verification.
- The load multipliers are returned sorted as by the dense solver: the
  positive ones first in increasing order, followed by the negative ones.

`freq` was verified not to be affected by either problem: with a negative
shift, the ordering of the shift-invert mode selects the lowest natural
frequencies for any shift, and the non-symmetric ARPACK routines do not call
`dsteqr`.

### Fixed

- Newton-Raphson advanced the load factor twice after each converged step and
  could finish without solving the final load factor, e.g. `initialInc=0.5`
  returned `increments == [0.5]`. Every analysis now finishes at a load factor
  of exactly 1.0, unless stopped by `minInc`.
- `modified_NR=False` updated the tangent stiffness matrix only every other
  iteration.
- The predictor of each Newton-Raphson step is scaled by the ratio between the
  new and the previous load increments. After a failed first step, the
  predictor was twice the linear solution.
- The divergence check stopped a step whenever the residual increased after
  the third iteration, which is normal in postbuckling. The divergence and
  too-slow checks now start after 5 iterations, comparing against the residual
  at the start of the step and the smallest residual of the last iterations.
- The line-search evaluated the internal forces twice per sub-iteration, and
  looped 20 times with a division by zero when the residual was already zero.
- Removed debug `print` calls from the Crisfield solver, which ignored
  `silent=True`.
- `Analysis.static()` raised a `ValueError` for an invalid `NL_method` without
  the name of the method in the message.

### Documentation

- New Sphinx documentation in `doc/`, built from the docstrings, with usage
  examples taken from the tests.
- Docstrings updated to describe the current implementation, including the
  callables required by `Analysis`, the default value of every parameter, and
  the return values of `static`, `freq` and `lb`.

### Tests

- Analytic problems with exact tangent stiffness matrices
  (`tests/analytic_problems.py`): hardening springs, a shallow von Mises truss
  and a linear problem.
- Newton-Raphson (`tests/test_newton_raphson.py`): quadratic convergence,
  load-factor bookkeeping, predictor, step cutting, non-monotonic residual and
  line-search.
- Arc-length (`tests/test_arc_length.py`): snap-through, snap-back, quadratic
  convergence, step cutting, `maxArcLength`, `silent` and independence of the
  units of the model.
- Linear buckling (`tests/test_linear_buckling.py`): sparse and dense solvers
  for a spectrum with load multipliers of mixed signs, fallback shift for a
  singular `K`, and regression tests with `K` and `KG` matrices of a plate and
  a cylinder from panels, saved in `tests/data/`, for the condensed solution,
  for `eigsh` alone (correct results or `RuntimeError`), for the verification
  of the eigenpairs, for the order of the load multipliers, and for null rows
  of `KG` and `K`.

## 0.3.1 (2026-04-09)

- Fixed the estimation of the shift `sigma` of the eigenvalue solvers in
  `freq` and `lb` for singular matrices, falling back to the default shift
  when the linear solution is singular or inaccurate.
- New `skip_null_cols` argument of `freq` and `lb`, to skip the removal of null
  rows and columns when the matrices are known to be non-singular.
- The dense solver of `freq` removes the null rows and columns in the same way
  as the sparse solver.
- `lb` no longer retries the sparse solver after removing the null rows and
  columns; they are removed before solving, unless `skip_null_cols=True`.
- Python 3.14 support.

## 0.3.0 (2026-03-26)

- `freq` and `lb` estimate the shift `sigma` of the shift-invert eigenvalue
  solvers from the matrices, instead of using a fixed value.
- Fixed the selection of `NL_method` in `Analysis.static()`, which compared
  strings with `is`.
- Packaging migrated from `setup.py` to `pyproject.toml`, requiring Python 3.8
  or newer.
- Continuous integration migrated from Travis CI to GitHub Actions, with
  workflows for tests, coverage, releases and publishing to PyPI.
- New test suite for `Analysis`, `solve`, `static`, `freq`, `lb` and
  `sparseutils`.

## 0.2.2 (2019-03-02)

- Release of 0.2.1 with an updated distribution.

## 0.2.1 (2019-03-01)

- Fixed the dense solver of `freq` (`sparse_solver=False`), which returned
  eigenvalues with the wrong sign, and which now also accepts dense arrays.

## 0.2.0 (2019-03-01)

- **Breaking:** `freq` returns the eigenvalues `lambda**2` of
  `([K] + lambda**2 [M]){u} = 0`, with `lambda**2 = -omega**2`, instead of the
  natural frequencies `omega`.
- **Breaking:** removed the `reduced_dof` argument of `freq`.

## 0.1.0 (2018-06-23)

- First release, with the solvers originally in `compmech.analysis` and
  `compmech.sparse`: linear static analysis (`static`, `solve`), linear
  buckling (`lb`), frequency analysis (`freq`), non-linear static analysis
  with the `Analysis` class using the Newton-Raphson and the arc-length
  methods, and utilities for sparse matrices (`sparseutils`).
