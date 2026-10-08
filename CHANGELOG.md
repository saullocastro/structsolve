# Changelog

## 0.6.1 (2026-10-08)

Includes the changes first published as 0.6.0, with the support of Pyodide.

### Pyodide support

`structsolve` runs in the browser with Pyodide (WebAssembly): the wheel
published on PyPI is pure Python and only requires NumPy and SciPy, which
Pyodide distributes, i.e. `await micropip.install("structsolve")`.

- The whole test suite runs in Pyodide, in the new GitHub Actions workflow
  `pyodide.yml`: the wheel built from the repository is installed with
  `micropip` in Pyodide under Node.js (`ci/pyodide/run_tests.mjs`, with
  the version of Pyodide pinned in `ci/pyodide/package.json`).
- `test_eig_complex_mkl_crash_regression` is skipped in Pyodide, which
  cannot start the subprocess of the test and whose SciPy is not linked to
  MKL.
- New classifier `Environment :: WebAssembly :: Emscripten`.
- Installation in Pyodide in the documentation.

### Eigenvalue solvers: sparse first, dense only as a last resort

`lb` and `freq` will run in the browser through Pyodide (WebAssembly), where
dense LAPACK is single-threaded and unoptimized: e.g. the dense `eig` of a
1,536-dof follower load problem takes 38 s there, against 0.3 s for a sparse
solver. Natively too, the dense paths are slow for follower loads, whose
load stiffness couples almost all the dofs, so that the condensation of `lb`
keeps almost all of them. The eigenvalue solvers now rely on the sparse
solvers, robustly: a verified result, or an explicit error, never a silently
wrong answer.

- **Solver policy.** With `sparse_solver=True`, the default, `lb` and `freq`
  always run the sparse solvers first, for every problem size, symmetric or
  unsymmetric, and every result is verified. Only when all the sparse
  solvers failed (verification, ARPACK error or no convergence, singular
  factorization, no positive real load multiplier...) a dense solver is used,
  if the dense problem has at most `max_dense_size` dofs, with a
  `DenseFallbackWarning` (new, `structsolve.DenseFallbackWarning`) giving
  the reasons of the failures; the dense result is verified as well.
  Otherwise a `RuntimeError` lists every attempted solver and its failure.
- **ARPACK and Intel MKL.** The `dsteqr` of Intel MKL 2024.2.0 to 2025.0.0
  returns wrong eigenvectors for tridiagonal matrices larger than 32 x 32,
  which ARPACK's symmetric drivers (`eigsh`) use with the size `ncv`. The
  failures start exactly at `ncv=33`, in every `eigsh` mode. The new module
  `structsolve.arpackutils` caps `ncv` at 32 when a functional probe of
  `dsteqr` fails, or when SciPy's LAPACK is MKL and its version is unknown
  or within the faulty range (read with the optional `mkl-service` or
  `threadpoolctl`). More than 15 eigenpairs are then computed in passes of
  at most 15 with deflation, since ARPACK hardly converges with `k` close to
  `ncv`. The cap can be overridden with `arpackutils.ARPACK_MAX_NCV` or the
  environment variable `STRUCTSOLVE_ARPACK_MAX_NCV` (`0` lifts it). It is
  not applied without MKL, e.g. in Pyodide, where a larger `ncv` is faster.
  `eigs` (non-symmetric drivers) is not affected. `freq(symmetric=True)`,
  added in 0.5.3, was affected and is fixed as well: with MKL 2025.0.0 it
  silently returned wrong frequencies, e.g. a lowest `omega**2` 1.3 and 12
  times too high for a plate and a cylinder of panels.
- **`lb`, symmetric matrices**: `eigsh` in Cayley mode with the capped `ncv`,
  retried with a 10 times larger shift; the dense condensed `eigh` is the
  last resort. The inertia check is stronger: the number of negative pivots
  of `K + s KG` (SuperLU without row interchanges, Sylvester's law of
  inertia) must equal the number of positive load multipliers found below
  `s`, slightly below the highest one returned, i.e. none of them is
  missing, not only the lowest one. The previous check is kept when a row
  interchange makes the count unknown.
- **`lb`, unsymmetric matrices**: the sparse fallback of 0.5.3, `eigs` on
  `K^-1 KG`, returned the multipliers of smallest modulus. With follower
  loads, e.g. a reversed (internal) pressure, none of them may be positive
  real, and `lb` silently returned negative or complex multipliers. It is
  replaced by a search of shifts along the positive real axis: `eigs` on
  `(K + s KG)^-1 KG` finds the multipliers nearest each shift `s`, the
  union of the discs of the steps covers `(0, lambda_cr)`, and a parity
  check of the sign of `det(K + 0.9999 lambda_cr KG)` confirms an even
  number of real multipliers below `lambda_cr`. The condensed dense `eig` is
  the last resort; when it finds no positive real multiplier either, i.e. no
  divergence load exists, its multipliers are returned with a
  `RuntimeWarning`.
- **`freq`**: new argument `max_dense_size=2000`. A failed verification of
  the sparse solver (residual, and for `symmetric=True` the inertia of
  `K - s M`, i.e. no lower mode missing) or an ARPACK error leads to the
  dense fallback, instead of a warning only. The symmetric dense fallback
  solves `M u = (1/omega**2) K u`, accurate for the lowest frequencies.
- The ARPACK starting vector is fixed, results are reproducible.
- In Pyodide, `gc.collect()` is called after each ARPACK call
  (`arpackutils.release_memory`): SciPy's `eigs` and `eigsh` keep the SuperLU
  factorization of the shift-invert mode in reference cycles, which exhausted
  the WebAssembly heap, e.g. `MemoryError` at the 10th `freq` call on a
  1,012-dof model.
- In Pyodide (`sys.platform == 'emscripten'`) the explicit dense `eig` uses
  the real LAPACK drivers, the complex ones are only needed against MKL.
- The static and non-linear solvers were audited: they only use sparse LU
  factorizations, no dense operation on full-size matrices.

### Behavioural changes

- `sparse_solver=True` now always means that the sparse solver runs first,
  whatever the size of the problem. In 0.5.3, `lb` solved the condensed
  problem with a dense solver first when it had at most `max_dense_size`
  dofs, i.e. most panels models. These problems now take the sparse path:
  natively, small and medium symmetric models can be slower than with the
  dense condensed solution, large models, follower loads and the browser
  are much faster.
- `max_dense_size` keeps its default `2000` but is now the maximum size of
  the dense **fallback** problem; `0` disables the fallback.
- A dense solution only happens as a fallback, with a
  `DenseFallbackWarning`.
- Where `lb` silently returned only negative or complex multipliers for
  unsymmetric matrices, it now finds the positive real one, falls back to
  the dense solver, or raises a `RuntimeError`.
- For unsymmetric matrices, when the search needed several shifts, the
  returned eigenpairs are the `num_eigvalues` multipliers nearest the final
  shift, sorted with the lowest positive real one first, not the ones of
  smallest modulus.
- `freq(sparse_solver=True)` raises a `RuntimeError` instead of a warning
  when its eigenpairs fail the verification and the dense fallback is not
  possible or fails too.
- `sparse_solver=False` is unchanged: the explicit full dense solution,
  returning all the eigenvalues, without trying the sparse solvers.

### Tests

- `tests/test_arpackutils.py`: the detection of MKL mocked both ways, the
  override, the `ncv=32`/`ncv=33` boundary, the passes, `eigs` unaffected,
  `freq(symmetric=True)` with a faulty `dsteqr`.
- `tests/test_sparse_first.py`: spies showing that the sparse solver runs
  first for any `max_dense_size`, also for small problems, and that no dense
  solver is called when it succeeds; the explicit dense path; the dense
  fallback with its warning and the `RuntimeError` without it; the cap
  disabled with a faulty `dsteqr`, where the dense fallback gives the
  correct result.
- Follower load regression tests with the cantilevers of Schweizerhof and
  Ramm (1984) from panels (new `tests/data/s1B1_m14n8_*`,
  `tests/data/s3B2_m14n8_*`), original and reversed, and harder mixed
  spectra for the symmetric solver.
- The test dependencies `buckling` and `composites` are removed: the
  hierarchical Legendre functions of Bardell (`vecf`, `vecfxi`, `vecfxixi`)
  and the stiffnesses of an isotropic plate (`isotropic_plate`) of the Ritz
  models of the tests are in the new `tests/ritz.py`, with NumPy only. The
  test extra `.[test]` no longer requires a pre-release (`buckling` is only
  published as `2026bN`) nor a compiled package (`composites`), whose wheels
  are not available for every platform, e.g. Pyodide.

## 0.5.3 (2026-10-06)

### New: configuration-dependent loads and unsymmetric eigenproblems

- The non-linear solvers pass the current load factor as `inc=lbd` to the
  callables `calc_fint`, `calc_kC` and `calc_kG` that declare a keyword
  argument `inc`, such that configuration-dependent loads, e.g. follower
  pressures, can enter the residual and the tangent stiffness matrix. The
  callables that do not declare it are called exactly as before (a
  `**kwargs` does not count). See the new module `structsolve.callbacks`.
- The arc-length methods (Riks and Crisfield) use `calc_fext(inc=1., c=c)`,
  the load vector of the current configuration, i.e. `dR/dlbd`, in the
  tangent predictor and in the bordered solutions, when `calc_fext` declares
  `c`. This restores the quadratic convergence of the Riks method for such
  loads.
- `lb(..., symmetric=None)`: with `None`, `K` and `KG` are checked with the
  new `linear_buckling.is_symmetric` (`||A - A^T|| <= 1e-10 ||A||`); `True`
  keeps the previous solvers without the check, `False` forces the solvers of
  unsymmetric matrices: `scipy.linalg.eig` on the condensed problem (keeping
  the dofs with a non-null row or column of `KG`), `scipy.sparse.linalg.eigs`
  on `K^-1 KG`, or `scipy.linalg.eig` on the full problem with
  `sparse_solver=False`. The residual of the eigenpairs is verified, the
  inertia check requires symmetric matrices and is skipped. Eigenvalues with a
  relative imaginary part below `1e-8` are returned as real, otherwise the
  complex ones come last. The arrays are real when the returned eigenpairs
  are real, even if other eigenvalues of the solver are complex, and a solver
  that raises an error is skipped, as one that fails the verification.
- The solvers of unsymmetric matrices in `lb` and `freq` call
  `scipy.linalg.eig` with complex matrices, i.e. the LAPACK drivers `zgeev`
  and `zggev`: with Intel MKL 2025.0.0 the real drivers `dgeev` and `dggev`
  crash the Python process for some matrices when computing the
  eigenvectors, with any number of threads.
- The Newton-Raphson and arc-length solvers already used LU factorizations
  (`spsolve`, `splu`), valid for unsymmetric tangent stiffness matrices.
- `Analysis.static(NLgeom=False)`: when `calc_fext` declares `c`, the
  geometrically linear problem with the load in the current configuration,
  `K0 c = F(c)`, is solved by load correction iterations, `c += K0^-1 (F(c) -
  K0 c)`, up to `relTOL`; for a follower load affine in `c` this is `(K0 +
  Kf) c = F0`. A warning is issued when they do not converge, beyond the
  critical load of the linearized problem. Callables without `c` are solved
  as before.
- `solve`, `remove_null_cols`: a degree of freedom is removed only when both
  its row and its column of the first matrix are null, such that an
  unsymmetric matrix never loses an equation with a non-null row; unchanged
  for matrices with a symmetric pattern. `static(K, fext)` documents
  unsymmetric `K`, e.g. `K0 + kCfollower`.
- `freq(..., symmetric=False, check_rtol=1e-6)`: the default keeps the
  general solvers (`eigs`, `eig`), valid for unsymmetric `K`, e.g. with the
  load stiffness of follower loads or aerodynamic matrices; `symmetric=True`
  uses `eigsh` (shift-invert) or `eigh`, with real results, and `None`
  selects them with `is_symmetric`. The relative residual of the eigenpairs
  is checked, with a warning above `check_rtol`. The Notes document the
  kinetic criterion (flutter: complex `lambda**2`). The printed eigenvalues
  are formatted as complex numbers when complex.

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
