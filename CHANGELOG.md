# Changelog

## 0.10.0 (2026-11-03)

### Robustness of the arc-length corrections

- New `Analysis.arc_length_line_search`, default `False`: the whole
  correction `(dc, dlbd)` of the arc-length methods is multiplied by the
  first factor of `arc_length.LINE_SEARCH_ETAS = (1., 0.5, 0.25)` that
  decreases the norm of the residual of the convergence criterion, or by
  the factor of the smallest norm when none does; the internal force vector
  of the accepted point is reused, such that a full correction costs no
  additional evaluation. `Analysis.line_search` still applies to
  Newton-Raphson only.
- New `Analysis.early_divergence`, default `False`, for Newton-Raphson and
  the arc-length methods: a step is stopped, and its increment cut, as soon
  as its relative residual exceeds `EARLY_DIVERGENCE_MAX_RREL = 1e3`, or,
  after `EARLY_DIVERGENCE_NUM_ITER = 3` iterations, its residual norm
  exceeds `EARLY_DIVERGENCE_FACTOR = 10` times its minimum in the step,
  instead of waiting for `NUM_ITER_BEFORE_DIVERGENCE_CHECK = 5` iterations
  and `DIVERGENCE_FACTOR = 1e3` times the first residual. New argument
  `Rrel` of `newton_raphson._check_divergence`. These are the criteria of
  the Riks analyses of a tow-steered cylinder for the DCSE website.
- New `tests/test_arc_length_robustness.py`. On the von Mises truss at 1.5
  times its limit load with a spring `ks = 1` and `initialInc = 0.6`, the
  line search avoids the step cut of the Riks method (7 instead of 10
  steps, 49 instead of 56 evaluations of `calc_fint`); with Crisfield's
  method it does not reduce the cuts and costs more evaluations on these
  problems.

### Indirect displacement control

- New `Analysis.NL_method = 'displacement_control'`, with the new
  attributes `Analysis.control_vector`, the vector `psi`, and
  `Analysis.control_scale`: the arc-length constraint is replaced by the
  linear constraint `psi^T Dc/s_ref = Ds`, with `s_ref = control_scale`,
  by default `|psi^T u_ref|`, the controlled quantity in the linear
  solution for a load factor of 1, such that `Ds = Dlbd` in the linear
  regime. The correction of the load factor is `dlbd = (s_ref Ds - psi^T
  (Dc + dc_R))/(psi^T dc_q)`. `psi` a unit vector controls one degree of
  freedom, e.g. the end shortening; a buckling mode controls its amplitude,
  with `control_scale` when its linear response is null. It shares the
  implementation of the arc-length methods, `arc_length._solver_arc_length`
  with `method='displacement_control'`, new module
  `structsolve.displacement_control`; `initialInc`, `minInc`, `maxInc` and
  `maxArcLength` are in units of `s_ref`, the first step being
  `initialInc`, and the line search and the early divergence check
  apply. The path is traced as long as the controlled
  quantity increases along it.
- New `tests/test_displacement_control.py`: on the von Mises truss loaded
  through a soft spring at 1.5 times its limit load, with a snap-through in
  the load and a snap-back in the displacement of the loaded point, the
  control of the apex deflection traces the path of the Riks method to the
  load factor 1 (14 steps, 32 evaluations of `calc_fint`, against 103 with
  the Riks method at the same `initialInc = 0.05`); the control of the
  loaded point misses the snap-back.

### Weights of the degrees of freedom in the arc-length norm

- New `Analysis.arc_length_weights`, default `None` (all 1): non-negative
  weights `w` of the degrees of freedom in the products of displacements
  of the arc-length constraint, `a^T diag(w) b`, also in the scale
  `u_ref^T diag(w) u_ref`, e.g. 1 for the translations and 0 for their
  derivatives or the rotations when the unknowns mix them. With Crisfield's
  convergence criterion, the analysis is then independent of the units of
  the degrees of freedom of null weight (new
  `tests/test_arc_length_weights.py`).

### Stability along the path: inertia of the tangent stiffness matrix

- New `Analysis.linear_solver`, `'splu'` (default) or `'pardiso'`, the
  factorization of the tangent stiffness matrix of the non-linear solvers,
  argument `linear_solver` of `static._factorize`. `'pardiso'` needs the
  shared library `mkl_rt` of Intel oneMKL (new optional dependency
  `structsolve[pardiso]`, the package `mkl`), called through the new
  module `structsolve.pardiso`, see below: the
  reduced matrix is scaled to a unit diagonal and its upper triangle
  factorized as real symmetric indefinite (`mtype=-2`); every solution is
  checked against its relative residual and, above `PARDISO_REFINE_TOL =
  1e-9`, improved by GMRES with the factorization as the preconditioner, a
  `RuntimeWarning` being issued above `PARDISO_WARN_TOL = 1e-6`. An
  unsymmetric matrix, e.g. with the load stiffness of a follower load, is
  factorized by `'splu'` instead, with a `RuntimeWarning`.
- The callable returned by `static._factorize` has the attributes
  `linear_solver`, `det_sign`, the sign of the determinant, from the
  diagonal of `U` and the permutations of SuperLU, `num_negative_eigvals`,
  the number of negative pivots of PARDISO (Sylvester's law of inertia),
  `None` with SuperLU, and `num_perturbed_pivots`, which make the count
  uncertain.
- New `Analysis.inertia`, default `False`: the tangent stiffness matrix is
  factorized at every converged increment of the Newton-Raphson, arc-length
  and displacement control solvers, the factorization being reused by the
  next step of the arc-length solvers, and `Analysis.num_negative_eigvals`,
  `Analysis.det_signs` and `Analysis.num_perturbed_pivots` record, per
  increment, its number of negative eigenvalues, the sign of its
  determinant and the number of perturbed pivots. New
  `newton_raphson._converged_state`.
- New `tests/test_inertia.py`: the count and the sign against dense
  eigenvalues and determinants, with null rows and a diagonal over 6
  orders of magnitude, and, along the snap-through of the von Mises truss,
  one negative eigenvalue exactly between the two limit points of the load;
  the same path with and without `inertia`. The tests of `'pardiso'` are
  skipped without oneMKL.
- New `Analysis.inertia_num_eigvals`, default `None`: with `inertia=True`,
  the number of negative real eigenvalues of a tangent stiffness matrix
  factorized by SuperLU, e.g. unsymmetric under a follower load, where the
  LU gives only the sign of the determinant, i.e. the parity of that
  number. The `k = inertia_num_eigvals` eigenvalues of smallest magnitude
  of the reduced tangent, on the space of the constraints, are computed at
  every converged increment by shift-invert at `sigma = 0` with ARPACK
  (`scipy.sparse.linalg.eigs` on the operator of the forward and backward
  substitutions of the LU of the converged state, no new factorization),
  new `static._smallest_eigvals` and `solve_kT.smallest_eigvals(k)` of the
  SuperLU factorizations, and those that are real (`|Im| <=
  INERTIA_REAL_RTOL = 1e-8` times the modulus) and negative are recorded in
  `Analysis.num_negative_eigvals`, so that `max_inertia_jump` and the
  detection of the crossings of `branch_switching` use them; the exact
  count of PARDISO is kept for symmetric matrices. It detects the
  divergence points, a real eigenvalue crossing zero; it does not detect
  flutter, a complex pair crossing the imaginary axis being no static
  critical point (their number among the `k` is logged). Negative real
  eigenvalues beyond the `k` are missed, and an odd number of them is
  reported, since the parity then differs from the sign of the
  determinant. New `newton_raphson._eigvals_inertia`; `max_inertia_jump`
  now requires `linear_solver='pardiso'` or `inertia_num_eigvals`.
- New `tests/test_inertia_unsymmetric.py`: the eigenvalues of ARPACK
  against `numpy.linalg.eigvals`, with null rows and with constraints
  (`Z^T K Z`); along the path of a non-normal unsymmetric tangent whose
  two real eigenvalues cross zero at `lbd = 0.27` and `0.63` and a complex
  pair at `0.47`, the count goes 0, 1, 2 with Newton-Raphson, Riks and
  Crisfield, as the dense eigenvalues, while the sign of the determinant
  returns to +1; `max_inertia_jump = 1` separates two crossings at `0.25`
  and `0.27`; the parity warning; on the symmetric von Mises truss the
  count of SuperLU equals the inertia. Cost on hpc12 (8 threads), a grid
  of 40000 hardening springs under a follower load with an unsymmetric
  load stiffness: the ARPACK count took 1.1 s with `k = 6` and 2.8 s with
  `k = 20` per converged increment, against 0.23 s for the SuperLU
  factorization, its 20 smallest eigenvalues being clustered in [10.04,
  12.39]; the Newton-Raphson analysis took 2.2 and 2.8 s per increment instead of
  1.1 s, the Riks analysis 2.2 and 2.9 s instead of 1.1 s.

### Linear constraints of the solution

- New `Analysis.constraints`, default `None`: the matrix `C` of shape
  `(n, m)`, a 1-D array for one constraint, of the linear constraints
  `C^T c = 0`, e.g. an inertia relief or multipoint constraints, in the
  linear static analysis and in the Newton-Raphson, arc-length and
  displacement control solvers. Every factorization of the stiffness
  matrices is bordered, `[[0, C^T], [C, K]]`, the border first, new
  argument `constraints` of `static._factorize`, with both linear solvers,
  such that every increment satisfies the constraints and Newton-Raphson
  keeps its quadratic convergence. The rows of the null rows and columns of
  the stiffness matrix are removed from `C`, and the constraints left
  without a non-zero term are dropped. The convergence criteria and the
  line searches measure the residual without the reactions of the
  constraints, its projection orthogonal to the columns of `C` in the
  inner product of the criterion, new `newton_raphson._residual_projector`.
  The inertia, `Analysis.inertia`, is that of the tangent on the space of
  the constraints: the number of negative pivots of PARDISO minus `m`, and
  the sign of the determinant times `(-1)**m` (Gould 1985).
- New `tests/test_constraints.py`: a multipoint constraint against the
  elimination of the constraint, with its reaction; the inertia relief of a
  chain of hardening springs without support under a self-equilibrated
  load, whose stiffness matrix is singular without the constraint; the
  quadratic convergence; the bordered factorization against the projected
  tangent `Z^T K Z`.
- New `Analysis.constraint_values`, default `None` (all null): the vector
  `d` of shape `(m,)` of the non-homogeneous constraints `C^T c = lbd d`
  along the non-linear path and `C^T c = d` in the linear static analysis,
  e.g. a displacement or a rotation prescribed in proportion to the load
  factor, with or without external load. `solve_kT` of `static._factorize`
  has the new optional argument `d`, the right-hand side of the border,
  `[[0, C^T], [C, K]] [mu; x] = [d; b]`. Since the constraints are linear
  in `(c, lbd)`, the solutions of the load vector, `dc_q`, the linear
  solution `u_ref` of the arc-length scale and the first Newton-Raphson
  predictor have the border `d`, and those of the residual a null border:
  every predictor (tangent, displacement control, generalized
  displacement control, the jump to `final_load_factor`, the restart) and
  every correction satisfy `C^T Dc = Dlbd d`, while the eigenvector
  predictor of the branch switching, at constant load, keeps a null
  border. Under a prescribed displacement alone, `fext = 0`, `u_ref` is not
  null and the reactions of the constraints, in the internal force vector,
  are the reference of the convergence criteria, both `'crisfield'` and
  `'relative'`. A constraint dropped with the null rows and columns must
  have a null value, otherwise `ValueError`; the values require
  `constraints`, one finite value per constraint. New
  `newton_raphson._constraint_values`.
- New `tests/test_constraint_values.py`: the bordered solutions; the
  linear chain of 5 springs with a prescribed end displacement, `u_i =
  delta i/n` and the reaction `k delta/n`, then with a load; the hardening
  springs driven only by a prescribed end displacement with each solver
  and both linear solvers, against the dense solution of the free degrees
  of freedom at every increment, and with a load and a prescribed relative
  displacement with both convergence criteria; modified Newton-Raphson;
  the branch switching of a column under a prescribed axial displacement,
  against its bifurcated branch `v**2 = (EA lbd delta - k)/(EA/2 + beta)`;
  the restart; null values bitwise the homogeneous constraints; the
  validation.
- New `tests/test_constraint_values_load_factor.py`: `constraint_values`
  together with a `calc_fext` that depends on the load factor, the
  springs of `tests/test_load_factor_dependent_load.py` whose node 3 is
  prescribed by the null-space method, `x3 = 1.5 lbd`, under a load `80
  lbd` on the node 2, with the node 1, `x1 = -0.5 lbd`, or the relative
  displacement `x2 - x1 = -0.5 lbd` prescribed by the constraint: every
  increment of Newton-Raphson, Riks, Crisfield, displacement control and
  generalized displacement control against the problem with both
  prescriptions eliminated by hand (one unknown, `brentq`) to `1e-8`, with
  `'pardiso'` and modified Newton-Raphson; no change of the solvers was
  needed.

### End of the non-linear analyses

- New `Analysis.stop_criterion`, default `None`: a callable
  `stop_criterion(an)` called after every converged increment of the
  Newton-Raphson, arc-length and displacement control solvers, with its
  load factor, solution and inertia already recorded; the analysis stops
  when it returns `True`. New module `structsolve.stopping` with
  `beyond_limit_point(num_increments)`, which stops a number of increments
  beyond the first limit point of the load, and `value_reached(func,
  value)`, e.g. a displacement measure.
- New `Analysis.final_load_factor`, default `1.`: the load factor at which
  the non-linear analyses finish, exactly; with the arc-length and
  displacement control solvers it may be `None`, the path being then
  traced beyond any load factor until `stop_criterion`, `maxArcLength`,
  `MAX_NUM_STEPS` or `minInc` stops it, for post-buckling paths whose load
  may exceed the reference load. New `tests/test_stopping.py`.

### Critical points along the path

- New `Analysis.max_inertia_jump`, default `None`: with `inertia=True` and
  `linear_solver='pardiso'`, a converged increment whose number of negative
  eigenvalues changed by more than this value is repeated with a smaller
  increment, as a failed one, in all the non-linear solvers. With `1` the
  eigenvalues cross zero one at a time. On the DOE09 cylinder (case 510)
  the tangent predictor jumped from 0 to 4 negative eigenvalues in one
  step at 4615 N, and a secant predictor (the previous increment scaled
  to the arc length), tried and not kept, reached 31-34 of them.
- New `Analysis.branch_switching`, default `0`: with `inertia=True`, the
  arc-length methods switch to the bifurcated branch at the first
  `branch_switching` bifurcation points crossed. A crossing is detected by
  the count of negative eigenvalues, or by the sign of the determinant with
  `'splu'`. It is classified as a bifurcation when the load varies in the
  same direction along the tangents before and after it. The next step
  starts along the critical eigenvector, computed by a subspace iteration
  with the factorization already available (`_critical_eigenvector`,
  restricted to `Analysis.constraints`), and is corrected with the
  spherical constraint. The detection and the switch along the critical
  eigenvector follow Wagner and Wriggers (Engineering Computations 5(2),
  103-109, 1988), without their location of the critical point by
  bisection. The new attribute
  `Analysis.branch_points` records the indices of the switched states.
  Tested on supercritical and subcritical symmetric bifurcations, whose
  branches `lambda P0 = k + beta v^2` are traced to 1e-6, and on the von
  Mises truss, whose limit points are not switched.
- New `Analysis.arc_length_sign_rule`, default `'inner_product'` (the rule
  of the previous versions), or `'determinant'`: with `inertia=True`, the
  load factor increases while the determinant of the tangent stiffness
  matrix has the sign of the one at the origin, and decreases otherwise.
  The path then cannot turn back along a branch without crossing a
  critical point. On the DOE09 cylinder the Crisfield run did exactly that
  with the inner-product rule: it went round a loop of unstable branches
  at about 4600 N and then went back along the fundamental path through
  the origin into tension, all with an even number of negative
  eigenvalues.
- `Analysis.branch_switching` also with
  `NL_method='displacement_control'` and
  `'generalized_displacement_control'`: the switching step is the one of
  the arc-length methods, along the critical eigenvector, corrected with
  the spherical constraint, whose radius is the length of the last
  increment in the arc-length norm times the growth of the step length
  since then (the arc length of the Riks and Crisfield methods). The next
  step starts along the tangent in the direction of the switching
  increment: the displacement control chooses the sign of its step
  accordingly, the generalized displacement control restarts its stiffness
  parameter from the current tangent, `GSP = <dq1, dq1>/<dq, dq>`, with the
  corrections orthogonal to it, as at its first step. The symmetric
  bifurcations of the tests are switched and their branches traced to
  1e-6 with both methods, for the steps 0.07, 0.13 and 0.17 and the rising
  and falling branches. On the DOE09 cylinder (80 increments) the
  generalized displacement control with `branch_switching=1` switched at
  4533.3 N, where the count fell from 4 to 2, and unloaded on the
  bifurcated branch to 4432.5 N with 1 negative eigenvalue (456 s), where
  without switching it rose again to 4504.2 N with 4.
- `Analysis.branch_switching` also where the number of negative eigenvalues
  decreases, e.g. a column unstable at the origin that the axial tension
  stabilizes, as in the algorithm of Wagner and Wriggers (p. 106, step
  (c): the number of negative diagonal elements changes); the eigenvector
  is then that of the smallest positive eigenvalues, the first that became
  positive. With `'splu'` the change of the sign of the determinant was
  already detected in both directions, with `'pardiso'` only the
  increases were.
- New `Analysis.bifurcation_pinning`, default `0`: a step of the arc-length
  or displacement control methods whose iterations failed this number of
  times in a row is repeated with the amplitude of the critical eigenvector
  held (`phi^T dc = 0` added to the constraints for that step), when it is
  orthogonal to the load (`|phi^T q| <= PIN_MAX_COS |q|`, `1e-3`); a step
  accepted without correction does not reset the count. Found with the
  stall of the DOE09 cylinder (case 510, 57000 degrees of freedom) with
  `max_inertia_jump=1`, whose mechanism is: at 4526.7 N the path reaches
  a bifurcation point whose eigenvector is orthogonal to the load to
  5.6e-8 (cosine, from the tolerance 1e-6 of the eigenvector of the
  2-wave perturbation, the problem being otherwise symmetric), with five
  other eigenvalues of the scaled tangent within 2e-5 of zero; the Riks
  correction of the load factor is then `-r_phi/q_phi`, the ratio of the
  components along the eigenvector of the residual left by the tolerance
  and of the load, about -9e-4 at every step length down to 1e-5, such
  that the iterations diverge (the tangent was verified against finite
  differences, the solutions to 1e-11); the shorter steps are accepted
  at their predictor, at the relative residual 1e-4 of the tolerance, the
  critical eigenvalue going from -9.3e-7 to -5.0e-8 and +4.8e-9 and the
  increments along its eigenvector. Neither the sign rule (the restart
  from the state before the stall with `'determinant'` stalls at the same
  point), nor the line search, nor the relative tolerance 1e-6 (no step
  converges), nor `branch_switching` (a limit point by the direction of
  the load) passes it; the inertia-jump check is not involved (no step
  was rejected by it there). With `bifurcation_pinning=3` the 80
  increments of the run pass the stall with three pinned steps: limit
  point at 4528.4 N, unloading to 4517.6 N at the 80th increment, 713 s
  against 769 s without, the path identical before the stall. With 160
  increments (8 pinned steps, 1383 s) the path unloads to 4420 N, follows
  a stable branch (0 negative eigenvalues, max |w| from 0.73 to 0.99 mm)
  up to 4540.4 N, and after two more critical points rises to 4615.2 N,
  the region of the limit load of the default Riks run (4638.2 N, which
  jumped from 0 to 4 negative eigenvalues in one step), the eigenvalues
  crossing zero one at a time up to 4 negative ones.
- `_critical_eigenvector` doubles its subspace, up to `EIGVEC_MAX_SIZE =
  64` vectors, until it holds the `num_neg` negative eigenvalues counted by
  the factorization: positive eigenvalues closer to zero, e.g. of weakly
  stiff in-plane terms of a Ritz model (about 1.5e-4 along the whole path
  of the benchmark below, in the scaled unknowns), filled the `num_neg + 2`
  vectors, and the eigenvector returned was not the critical one. With
  `'splu'`, without a count, at least one negative eigenvalue is looked
  for when the determinant is negative: the perfect cylinder of the
  benchmark below then switches with 10 x 10 and 12 x 12 terms as with
  `'pardiso'` (0.8270 and 0.8267), where before the path stayed on the
  fundamental branch.
- New tests in `tests/test_branch_switching.py`.

### Generalized displacement control

- New `NL_method='generalized_displacement_control'`, the method of Yang
  and Shieh (AIAA Journal 28(12), 2110-2116, 1990), implemented in the
  arc-length solver:
  - the load increment is scaled by the square root of the generalized
    stiffness parameter, whose sign reverses the load at the limit points;
  - the corrections are orthogonal to the tangent of the previous step,
    in the norm of `arc_length_weights`.
  - With the default weights, these are Eqs. (30), (33) and (34) of the
    paper. The load reverses relative to the previous step when the
    parameter is negative (the paper's step 2d).
- The step is a load factor and is never increased beyond `initialInc`.
  Growing it as the arc length is grown made the von Mises truss jump
  across its limit point, from a load factor of 0.63 to -0.62.
- The equations were written from the description of the method, not
  checked against the paper.
- New module `structsolve.generalized_displacement_control` and new
  `tests/test_generalized_displacement_control.py`.

### Restart

- After every converged increment, the new `Analysis.path_state` holds the
  state from which the analysis can continue: the solution, the load
  factor, the next increment and the directions of the predictors, as
  arrays and numbers that can be pickled. Set `Analysis.restart_state` to
  it to continue from that increment, e.g. with other options or in
  another job. This works for the Newton-Raphson, arc-length and
  displacement-control solvers.
- The restarted path equals the uninterrupted one to round-off. New
  `tests/test_restart.py`.

### Cost of the factorizations

- `newton_raphson._linear_solve`, used for the linear solution of the
  arc-length scale and of the first predictors, now uses
  `Analysis.linear_solver`. It always used SuperLU before: with
  `'pardiso'` on the DOE09 cylinder (57000 DOF), that one-off SuperLU
  factorization took 14.4 s.
- `sparseutils.remove_null_cols` finds the null rows and columns from the
  stored terms directly and skips the slicing when there are none. Its
  time halved on a 57000-DOF matrix, from 0.14 to 0.07 s.
- `static._is_symmetric` and the scaling of the matrix factorized by
  PARDISO avoid intermediate sparse products. The overhead of `_factorize`
  around the PARDISO factorization dropped from about 0.2 s to 0.15 s on
  the same matrix. The solutions are unchanged to 4e-17.
- On the cylinder, a profile on hpc12 put about 26 % of each tangent in
  the preprocessing of `_factorize` and 19 % in the conversions to sparse
  formats of the calling script. The element loop accounted for less than
  1 %.
- New `static._FactorizationCache`, created by `Analysis.static` for each
  non-linear analysis and passed to `static._factorize` as the new argument
  `cache` by every factorization of the tangent stiffness matrix: the work
  that depends only on the sparsity pattern is done once per pattern. The
  reduced and bordered matrix, its CSC form, the transposed terms of the
  symmetry check, the diagonal of the scaling and the scaled upper
  triangle of PARDISO are built from index arrays that gather the values
  of `kT`, and the analysis of PARDISO (reordering and symbolic
  factorization, phase 11) is reused: the next factorizations use its
  phase 22 instead of 12, with up to `static.CACHE_MAX_PARDISO_SOLVERS = 4`
  solvers that keep the analysis, since the factorizations of the
  converged state and of the iteration coexist. The pattern (shape,
  `indptr`, `indices`), the null rows and columns, found from the values at
  every call since explicit zeros do not count, and the constraints are
  checked at every call, and any change rebuilds the cache; the symmetry
  and the scaling are still computed from the values. The factorizations,
  solutions, inertia and perturbed pivots are bitwise those without the
  cache (tested with `MKL_NUM_THREADS=1`). `_factorize` without `cache`
  is unchanged; a matrix that is not `float64` or not in canonical CSR form
  is not cached. On a symmetric 57000-DOF matrix with 4.5 million terms,
  one PARDISO factorization took 0.40-0.44 s instead of 1.65-1.78 s, with
  or without one constraint (the phase 22 alone 0.20 s instead of 1.0 s for
  the phase 12), and the preprocessing of SuperLU 0.05 s instead of
  0.11 s, of about 1.1 s per factorization. A Newton-Raphson analysis of a
  grid of 40000 hardening springs took 2.8 s instead of 4.8 s with
  `'pardiso'`, its Riks analysis 3.7 s instead of 7.1 s, with SuperLU
  within 3 %. New `sparseutils._used_cols`.
- New `tests/test_factorization_cache.py`: bitwise the factorization
  without the cache, with null rows, explicit zeros, a border and
  unsymmetric matrices; the phases of PARDISO and the rebuilds when the
  pattern, the null rows or the constraints change; the Newton-Raphson,
  Riks, Crisfield and generalized displacement control analyses with
  constraints and `inertia`, and the Riks analysis beyond the limit point
  of the von Mises truss, bitwise those without the cache. Constraints
  given with duplicated terms are not cached.
- New module `structsolve.pardiso`: `PardisoSolver`, the interface to
  PARDISO of Intel oneMKL through `ctypes`, which calls the phases 12, 22
  (the numerical factorization with the analysis of the last phase 12),
  33, 0 and -1 directly and keeps a reference to the values of the
  factorized matrix only, for the iterative refinement of PARDISO.
  `mkl_rt` is given by the environment variable `STRUCTSOLVE_MKL_RT`,
  otherwise found by `ctypes.util.find_library`, in `lib` or
  `Library/bin` of the environment, or under it or the user site, and
  loaded once per process. New `tests/test_pardiso.py`.

### Benchmark: post-buckling of an axially compressed cylinder

- New folder `notebooks/`, as in `panels`: the notebook
  `notebooks/simo1990_cylinder_postbuckling.ipynb`, stored with its
  outputs, with the utilities and the command line of the convergence runs
  in `notebooks/simo1990_utils.py`, the test
  `tests/test_simo1990_cylinder.py` (skipped where `panels`, which is not a
  dependency of structsolve, is not installed; about 1.5 min) and the
  documentation page `ex_cylinder_postbuckling`: the cylinder of Simo, Fox
  and Rifai (1990), Section 6.3.4, Fig. 6.3.4, an `L/10 x 15 deg` section
  of `panels` 0.15.0 (`cylshell_clpt_donnell`), traced with the Riks method
  to twice the shortening of the linear buckling load. The imperfection is
  added exactly for the Donnell kinematics, `fint(c + c0) - fint(c0) -
  kG(N0) c`, checked against the gradient of the strain energy to 1.7e-10.
  With 12 x 12 terms the limit loads are 0.727 (10 % imperfection) and
  0.390 (50 %), against 0.715 and 0.411 read from the figure and 0.717 and
  0.402 for Stanley's points, unchanged in the third digit from 10 x 10 to
  14 x 14 terms; the snap-back, the plateau at 0.32 to 0.33 and the loads
  up to the shortening 0.04 agree within 0.02. The perfect cylinder with
  `branch_switching`, 8 x 8 terms, bifurcates at 0.827 (0.833 read) and
  follows Fig. 6.3.4 (d) to the plateau.
- Found with it: the count of negative eigenvalues of PARDISO
  (`linear_solver='pardiso'`, about 20 perturbed pivots) found the first
  critical point at the load 0.90 instead of 0.83, and
  `arc_length._critical_eigenvector`, whose subspace has the `num_neg + 2`
  eigenvalues closest to zero, returned an eigenvector of weakly stiff
  in-plane Ritz terms instead of the negative eigenvalue, such that the
  switch converged back on the fundamental path. Both fixed, see "Critical
  points along the path" and "PARDISO with perturbed pivots": with
  `linear_solver='pardiso'`, and with `'splu'` and the same paths, the
  perfect cylinder (unknowns scaled to a unit diagonal) now switches at the first
  critical point, 0.8273, 0.8270 and 0.8267 with 8 x 8, 10 x 10 and 12 x 12
  terms, where a dense eigenvalue analysis of the tangent has its first
  negative eigenvalue, and follows the bifurcated branch down to the
  plateau, 0.327, 0.326 and 0.326 (174 to 182 increments, 57 s, 189 s and
  428 s on one thread).

### PARDISO with perturbed pivots

- `static._factorize` with `'pardiso'`: when the factorization with the
  default parameters of PARDISO perturbs pivots, the matrix is factorized
  again with `PARDISO_FALLBACK_IPARM = {24: 1}`, the two-level
  factorization algorithm, and the factorization with fewer perturbed
  pivots is kept (`solve_kT.pardiso_fallback`); with the cache of a
  non-linear analysis the next factorizations of the pattern use these
  parameters directly. The dense tangent of the benchmark below (8 x 8
  terms, 114 degrees of freedom) had 18 to 20 perturbed pivots along the
  path, solutions of relative residual 0.4 to 1 (before GMRES) and one
  negative eigenvalue less than the dense eigenvalue analysis from the load
  0.827 to 0.893; with `iparm(24) = 1` no pivot is perturbed, the
  residuals are 1e-13 and the counts are those of the dense analysis at
  every increment, with 8 x 8 and 12 x 12 terms. Neither the perturbation
  `iparm(10)` (8, 13, 20), the scaling and weighted matching `iparm(11) =
  iparm(13) = 1`, nor `iparm(21) = 3` changed the perturbed pivots; the
  minimum degree ordering `iparm(2) = 0` fixed the benchmark but not a
  dense random indefinite matrix of the same size, which `iparm(24) = 1`
  fixed. The factorizations without perturbed pivots, e.g. all those of
  the DOE09 cylinder, are unchanged. New tests in `tests/test_inertia.py`,
  one with a dense matrix (34 perturbed pivots with MKL 2026.1, none with
  2026.2) and others that fake the perturbed pivots, for every MKL.
- `solve_kT.pardiso_fallback` is set by `_factorize` instead of being read
  from `iparm(1) = 1`, which MKL 2026.2 also sets with its default
  parameters: the fallback was reported for every factorization.

## 0.9.4 (2026-10-28)

### Python versions

- Python 3.9 is no longer supported, it reached its end of life in October 2025: Python 3.10 or newer is required, and the tests and wheels cover Python 3.10 to 3.14.

### Packaging

- Packaging: the license is given as the SPDX expression `BSD-3-Clause` with `license-files` (PEP 639), replacing the TOML table and the license classifier that setuptools deprecates, hence `setuptools>=77`; wheel is no longer a build requirement; link to the documentation in the project URLs.

### New default: Crisfield's convergence criterion

- The non-linear analyses of `Analysis`, Newton-Raphson, arc-length and
  the load correction iterations of the linear static analysis with a
  configuration-dependent load, check the convergence by default with
  Crisfield's criterion, `Analysis.convergence = 'crisfield'`: the residual
  and the reference force vectors are measured with the norm scaled by the
  diagonal of the linear stiffness matrix `calc_kC()`, over the free
  degrees of freedom, those whose row or column of `calc_kC()` is not
  null, and `zeta = ||R||_d/max(||lbd*Fext||_d, ||Fint||_d,
  ||dlbd*Fext||_d, ||lbd*dR/dlbd||_d) <= relTOL`, the third reference in
  the arc-length methods only, the fourth when `calc_fext` declares `lbd`.
  The same scaled norm is used by the divergence and too-slow checks and
  by the line search. The criterion does not depend on the units of each
  degree of freedom, e.g. translations and rotations, and ignores the
  residual at the constrained degrees of freedom, e.g. support reactions.
  The same implementation as `pyfe3d.solver`.
- The default `relTOL` is `1e-9`, instead of `1e-6`. The former check,
  the Euclidean norm over all the degrees of freedom, is kept with
  `Analysis.convergence = 'relative'`; `absTOL` is accepted with both. A
  null diagonal entry of `calc_kC()` at a free degree of freedom raises a
  `ValueError` with `'crisfield'`.
- New module `structsolve.convergence`: `crisfield_weights`, `scaled_norm`,
  `crisfield_zeta`, `check_convergence`, `CRITERIA`. New
  `newton_raphson._weights`, `_norm`, `_log_iteration`; argument `w` of
  `newton_raphson._check_convergence` and `_NR_iterations`. The log of the
  iterations prints `zeta` with `'crisfield'`.

### Round-off floor of the convergence criteria

- With both criteria, an iteration has also converged when the norm of its
  residual, `||R||_d` with `'crisfield'` and the Euclidean norm with
  `'relative'`, is not larger than `convergence.ROUNDOFF_FACTOR = 3` times
  the machine precision times the same norm of `|K_C0| |c|`, the product of
  the elementwise absolute values of `calc_kC()` and of the solution of the
  iteration, below which the residual cannot be computed. Like `absTOL`, it
  is an alternative to `relTOL`; `zeta` is unchanged, and the log of the
  iterations marks those converged at the round-off floor. With
  `relTOL = 1e-9`, the arc-length methods stopped on a low sinusoidal arch
  of Fung and Kaplan (NACA TN 2840, 1952) of 40 BeamC elements of pyfe3d at
  load factors of 0.026 to 0.032, close to the null load between the limit
  points, where the residual of the large self-equilibrated axial forces
  stagnates at 0.29 to 0.47 times `eps ||(|K_C0| |c|)||_d`, and with 160
  elements before the upper limit point; they now reach the load factor
  1.0. Over the test suite, the floor changes the verdict of no iteration
  besides those of the new unit test. With load vectors that depend on the
  load factor, `|K_C0| |c|` omits the prescribed part of the displacements:
  the floor is then lower, i.e. conservative.
- New `convergence.ROUNDOFF_FACTOR` and `convergence.roundoff_floor`; new
  keyword arguments `Kc_abs` and `full_output` of
  `convergence.check_convergence` and `newton_raphson._check_convergence`,
  whose defaults keep the former behaviour, `K0_abs` of `_NR_iterations`
  and `roundoff` of `_log_iteration`; new `newton_raphson._abs_matrix`.
  `ROUNDOFF_FACTOR = 0` disables the floor, as in
  `test_nr_unreachable_tolerance_stops_without_increments` and
  `test_displacement_control_without_lbd_does_not_converge`, which test the
  behaviour without it; new test `test_check_convergence_roundoff_floor`.

### Bug fixes

- `lb`: the negative load multipliers returned after the positive ones are
  those closest to zero, i.e. the critical load multipliers of the reversed
  load, instead of those of largest modulus, for the sparse and the dense
  solvers, symmetric or unsymmetric. The sparse solver of symmetric
  matrices computes them with `-KG` when fewer than `num_eigvalues`
  positive load multipliers exist, and verifies them with the same
  residual and inertia checks, such that a reference load without any
  positive load multiplier, e.g. a tensile load, no longer needs the dense
  fallback. The infinite load multipliers, e.g. of the null rows of `KG`,
  come last, and the sparse solver of symmetric matrices does not return
  them. The load multipliers of each sign are counted with the inertia of
  `K + s KG` before requesting them from ARPACK, which does not converge
  when it must return eigenvalues of the cluster of the null ones, e.g.
  for the tensile reference load of a plate. New
  `linear_buckling._eigsh_both_signs`, `_count_multipliers`,
  `_check_both_signs`, `_lb_order`, `_null_mu`, `NULL_MU_RTOL` and
  `INERTIA_COUNT_FACTOR`.
- `freq(sort=True)`, the default, sorts and filters `lambda2` together with
  the eigenvectors, instead of the eigenvectors only, such that
  `eigvecs[:, i]` is always the eigenvector of `lambda2[i]`. The eigenpairs
  with a natural frequency not larger than `1e-6` rad/s, e.g. rigid-body
  modes, are removed from both arrays, which may then have fewer than
  `num_eigvalues` entries.
- `freq(sort=True)`: the eigenpairs with the same rounded sort key, the
  real then the imaginary part of `omega_n` rounded to one decimal, are
  sorted by the exact `omega_n`, real then imaginary part, then by the
  exact `omega_n**2 = -lambda2`, instead of keeping the order of the
  solver. A double eigenvalue returned with a round-off difference in an
  order that depends on the LAPACK build, e.g. `omega_n**2 = 100` by the
  dense `eig()` with OpenBLAS, is returned in increasing order whatever
  the order of the solver. `omega_n**2` breaks the ties of `omega_n`
  after the square root, e.g. of `100` and `100` minus one unit in the
  last place, both `omega_n = 10.0`.
- `freq`: the residual of a rigid-body mode, whose `||K u|| + |lambda2|
  ||M u||` is not larger than `linear_buckling.RIGID_BODY_RTOL = 100 eps`
  times the matrix-norm scale `(||K||_F + |lambda2| ||M||_F) ||u||`, is
  relative to this scale, i.e. the backward error, see
  `linear_buckling._relative_residuals(rigid_rtol)`. The rigid-body modes,
  `lambda2 = 0` and `K u = 0` up to the round-off, whose relative residual
  `||K u + lambda2 M u||/(||K u|| + |lambda2| ||M u||)` was about 1, failed
  the verification of the sparse solvers and of the dense fallback,
  raising a `RuntimeError`.
- `lb` and the other eigenpairs of `freq` keep the relative residual of
  0.8.0, without any floor of its denominator. The floor of `1e-8` times
  the matrix-norm scale for every eigenpair, in the development version,
  loosened the check about 1000 times for models with stiff penalty
  springs, whose `||K||_F` is many orders of magnitude larger than
  `||K u||`: inaccurate eigenpairs of `eigsh` were accepted without
  warning, e.g. lowest load multipliers of blade-stiffened plates of
  `panels` with penalty connections from 1.3 % below to 0.26 % above those
  of the null-space method, which the penalty values approach from below;
  they are now rejected and computed with the dense fallback. New test data
  `tests/data/blade_stiffened_penalty_K.npz` and `_KG.npz`.
- `freq`: an eigenpair whose relative residual is larger than `check_rtol`
  passes the verification when its normwise backward error
  `||K u + lambda2 M u||/((||K||_F + |lambda2| ||M||_F) ||u||)` is not
  larger than `linear_buckling.BACKWARD_ERROR_RTOL`, the machine precision,
  i.e. when it is accurate up to the round-off (issue #14). The genuine
  eigenpairs with an eigenvalue close to zero of a nearly singular `K`,
  e.g. pre-stressed close to buckling, failed the verification of the
  sparse solvers and of the dense fallback, raising a `RuntimeError`
  unless a larger `check_rtol` was given: for a cantilever of `pyfe3d`
  pre-stressed to 0.9999 of its buckling load, `omega**2 = 0.0895` with
  relative residuals of 2.5e-6 to 9.1e-6 and backward errors of at most
  0.05 times the machine precision. The inaccurate eigenpairs of the
  penalty models above, with backward errors of 14 times the machine
  precision and more, are still rejected; `lb` keeps the relative residual
  only. New `linear_buckling._backward_errors` and `freq._failed_residuals`,
  and test data `tests/data/beam_cantilever_K.npz`, `_KG.npz` and `_M.npz`.
- `lb`, `freq` and `arpackutils.capped_eigsh`: the calls of ARPACK are
  reproducible with SciPy 1.17 or newer. When the Krylov subspace is
  invariant, e.g. with a nilpotent `K^-1 KG`, ARPACK draws new starting
  vectors from the argument `rng` of `eigs` and `eigsh`, by default from
  the entropy of the operating system: with the same `v0` the results
  differed between calls, e.g. `lb` on Ziegler's double pendulum raised a
  `RuntimeError` or returned a spurious load multiplier of about `1e33`
  at random. New `arpackutils.start_kwargs(n)`, the arguments `v0` and,
  when supported, `rng` with the seed `42`, used by every call of ARPACK.
- `lb` with unsymmetric matrices, sparse shift search: the eigenvalues
  `nu` of `(K + s KG)^-1 KG` found by `eigs` at each shift are null when
  `|nu| <= NULL_MU_RTOL max(max|nu|, ||(K + s KG)^-1 KG||)`, the norm
  estimated with one step of the power iteration from the starting vector
  of ARPACK, instead of `|nu| <= 1e-12 max|nu|`. When `K^-1 KG` has no
  non-null eigenvalue, e.g. nilpotent as with the follower load of
  Ziegler's double pendulum, all the `nu` are round-off and `max|nu|` is
  not a scale: spurious load multipliers of about `1e33` (SciPy 1.17.1)
  or `1e17` (SciPy 1.15.3) were found, and `lb(max_dense_size=0)` raised
  the expected `RuntimeError` only when the parity check rejected them,
  depending on the platform and, with SciPy < 1.17, whose ARPACK keeps its
  restart seed from one call to the next, on the calls made before in the
  same process. They are now all null, and the shift search raises "no
  finite load multiplier was found". New `linear_buckling._norm_estimate`.
- `lb`, dense solvers (`sparse_solver=False` and the dense fallback, of
  symmetric and unsymmetric matrices): an eigenvalue `mu` of
  `KG u = mu K u` is also null when `|mu| <= 10 eps ||KG||_F kappa`, `kappa`
  its condition number from the left and right eigenvectors, i.e. when
  the round-off does not determine it, instead of only when
  `|mu| <= 1e-12 max|mu|`. The round-off splits a defective null
  eigenvalue, a Jordan block of size `m`, into `m` eigenvalues of modulus
  about `eps^(1/m)`, e.g. `+-9.4e-10` for Ziegler's double pendulum, where
  `K^-1 KG` is nilpotent: `lb` returned the spurious load multipliers
  `+-1.07e9`, about `8.5e7` times the flutter load, instead of none. The
  null `mu` are set to 0 and their load multipliers are `-inf`, also in
  complex arrays; the dense fallback then finds no finite load multiplier
  and fails the verification. Well-conditioned eigenvalues are null only
  below `2.2e-15 ||KG||_F`; in the follower-pressure cantilevers of the
  tests, 6 and 22 more eigenvalues are null, all below `2e-9 max|mu|`, and
  the lowest positive real load multipliers do not change. New
  `linear_buckling._null_mu_pencil`, `NULL_MU_PERTURBATION_FACTOR`;
  argument `left` of `_eig_complex`.
- Arc-length solvers, `arc_length_riks` and `arc_length_crisfield`: a
  converged step whose increment points against the tangent predictor,
  i.e. a negative or null inner product of the arc-length constraint, is
  rejected and repeated with a shorter arc length. The previous converged
  point is at the distance of the arc length from the last one, and the
  iterations could converge to it and then trace the path backwards
  ("doubling back"), e.g. the von Mises truss at 10 times its limit load
  with `initialInc=0.2`, with both convergence criteria, and the hinged roof
  of Sze et al. (2004), `[90/0/90]`, thickness 12.7, third-order shear
  deformation theory, of the notebooks of `panels`, with Crisfield's
  criterion: the step through the limit point returned exactly to the
  converged point before the last one. Paths without such a step are
  unchanged.
- `lb`, `freq` and `arpackutils.capped_eigsh` no longer reset the registry
  of the warnings already shown: a warning of the caller under the
  `'default'` action, shown once per location, was shown again after each
  call, e.g. in a loop calling `lb`. Seven helpers wrapped
  `scipy.sparse.linalg.splu`, or `spsolve` in `freq`, in
  `warnings.catch_warnings()`, whose exit resets this registry of every
  module. `splu` of a CSC matrix issues no warning, and the blocks are
  removed; the shift of `freq` detects an exactly singular stiffness matrix
  with the `RuntimeError` of `splu` instead of the `MatrixRankWarning` of
  `spsolve`. New test `tests/test_warnings_registry.py`.

### Sign convention of the residual

- The residual is `R = Fint - lbd*Fext`, instead of `lbd*Fext - Fint`, in
  the code and the documentation, such that the tangent stiffness matrix is
  its Jacobian, `K_T = dR/dc`, the Newton-Raphson corrections solve
  `K_T dc = -R`, and `dR/dlbd = -Fext`; the load vector that
  `calc_fext(inc=1., c=c)` or `calc_fext(inc=1., c=c, lbd=lbd)` returns,
  unchanged, is `-dR/dlbd`. The same convention as `pyfe3d.solver` and
  `panels`. Each changed operation is the exact negation of the former one:
  `fint - fext` instead of `fext - fint` in
  `Analysis._linear_static_follower`, `newton_raphson._NR_iterations`
  (also in the line search) and `arc_length._solver_arc_length`, and
  `solve_kT(-R)` or `solve_k0(-R)` instead of `solve_kT(R)` or
  `solve_k0(R)`. Over the test suite, the load factors and the solution
  vectors of the 139 calls of `Analysis.static`, 2411 vectors, are bitwise
  identical before and after, the signs of the zeros included. The test
  helpers that compute residuals follow the same convention; the method
  `dR_dlbd` of `tests/test_load_factor_dependent_load.py` is renamed
  `load_vector`.
- **Breaking**, in the documentation of an argument only: the argument `R`
  of `convergence.check_convergence` and `convergence.crisfield_zeta`,
  as released in 0.9.2, and of `newton_raphson._check_convergence`, is
  `fint - fext` (`fint - fext_total`) instead of `fext - fint`. These
  functions use only norms of `R` and `max(abs(R))`, and they return the
  same values for both signs. No callable of `Analysis` receives or returns
  a residual, and no attribute stores one.

### Documentation

- The documentation workflow installs pandoc, which nbsphinx needs to convert the tutorial notebook; the documentation was not published because of it. The documentation is now also built, without being deployed, for every pull request.
- `structsolve.convergence`: the origin of the scaled norm is cited as the sources state it: Crisfield (1981, p. 58) attributes the scaling to Peano and Riccioni and uses it in his arc-length procedure, and Crisfield (1991, Sec. 9.5.4, Eq. 9.48) presents it as an alternative to the Euclidean norm of the residual, not as a recommendation; the dependence of the Euclidean norm on the units is no longer attributed to Crisfield (1991), who says that the scaling matrix could be used to give all parameters the same dimensions.
- `structsolve.convergence`: the round-off floor cites Higham (2002), Sec. 3.5, Eqs. (3.12) and (3.13), for the componentwise bound of the rounding error of a matrix-vector product.
- Notation: the basis of the null space of the constraints is `Z`, `c = Z c_r + c_p0 + lbd*c_p1`, instead of `T`, in `structsolve.callbacks`, the pages on boundary conditions and prescribed displacements, the tests and the 0.7.0 entry of this changelog; no name of the API changes, none contained the basis.
- Notation of the book in the docstrings, the user guide, the tutorial and the math of the tests, without code changes: vectors are bold lower case and matrices bold upper case, without braces or brackets, e.g. `\boldsymbol{c}`, `\Delta \boldsymbol{c}`, `\boldsymbol{K}_T` and `\boldsymbol{Z}` instead of `\{c\}`, `\{\Delta c\}`, `[K_T]` and `[Z]`; the force vectors and the residual keep their upper case, `\boldsymbol{F}_{ext}`, `\boldsymbol{F}_{int}` and `\boldsymbol{R}`, also in the derivatives, e.g. `\partial \boldsymbol{R}/\partial \lambda`, and in the scaled norms of Crisfield's criterion; explicit lists of components keep their braces; the transpose is `^\top` instead of `^T`. Three docstrings that now contain `\boldsymbol` became raw strings.
- Notation, without code changes: the permutation matrices of the sparse factorizations are `\boldsymbol{\Pi}`, `\boldsymbol{\Pi}_r` and `\boldsymbol{\Pi}_c`, `\boldsymbol{\Pi} \boldsymbol{A} \boldsymbol{\Pi}^\top = \boldsymbol{L} \boldsymbol{D} \boldsymbol{L}^\top` and `\boldsymbol{\Pi}_r \boldsymbol{A} \boldsymbol{\Pi}_c = \boldsymbol{L} \boldsymbol{U}`, instead of `\boldsymbol{P}`, `\boldsymbol{P}_r` and `\boldsymbol{P}_c`, in the docstrings of `_negative_pivots` and `_det_sign` of `structsolve.linear_buckling` (the attributes `perm_r` and `perm_c` of SuperLU are unchanged); the standard problem of the condensed dofs in the docstring of `_eig_condensed` is written with the Schur complement, `(\boldsymbol{K}_{bb} - \boldsymbol{K}_{ba} \boldsymbol{K}_{aa}^{-1} \boldsymbol{K}_{ab})^{-1} \boldsymbol{K}_{G_{bb}}`, as in `_eigh_condensed`, instead of `\boldsymbol{S}^{-1} \boldsymbol{K}_{G_{bb}}`, `\boldsymbol{S}` being the shape functions; the configuration-dependent load of `tests/test_follower_loads.py` and of the page on follower loads is written with the load stiffness matrix of the book, `\boldsymbol{K}_F = -\partial \boldsymbol{F}/\partial \boldsymbol{c} = \lambda \boldsymbol{K}_{F_{ref}}`, i.e. `\boldsymbol{F}(\boldsymbol{c}, \lambda) = \lambda (\boldsymbol{q}_0 - \boldsymbol{K}_{F_{ref}} \boldsymbol{c})`, instead of `\lambda (\boldsymbol{q}_0 + \boldsymbol{Q} \boldsymbol{c})`; the variable `Q` of the tests, `-\boldsymbol{K}_{F_{ref}}`, keeps its name.

### Tests

- The partial derivatives of the Ritz models of the tests are named with an underscore introducing the derivative: `vecfxi`, `vecfxixi` of `tests/ritz.py` become `vecf_xi`, `vecf_xixi`; in `tests/test_linear_buckling.py` and `tests/test_static_deflection.py`, `Swx`, `Swy`, `Swxx`, `Swyy`, `Swxy` become `Sw_x`, `Sw_y`, `Sw_xx`, `Sw_yy`, `Sw_xy`, `Sphixx`, `Sphixy`, `Sphiyx`, `Sphiyy` become `Sphix_x`, `Sphix_y`, `Sphiy_x`, `Sphiy_y`, the functions at `xi` and `eta`, `P_xi` and `P_eta`, become `Pxi` and `Peta`, and their derivatives `Px_xi`, `Pxx_xi`, `Px_eta`, ... become `Pxi_xi`, `Pxi_xixi`, `Peta_eta`, ...; the quadrature weights `wxi` and `weta` keep their names. The same names as in the copies of these models in the tests of pyfe3d.
- `tests/test_arc_length_fung_kaplan.py`: both arc-length methods, with both convergence criteria, on the modal model of a low sinusoidal arch under a sinusoidal load of Fung and Kaplan (NACA TN 2840, 1952), harmonics 1 and 2: the symmetric path, their Eq. (23), through both limit points, whose loads are those of their Eq. (25) within a tolerance given by the step size, the two null loads and the bifurcation point of their Eq. (31), before the limit point only when `lambda_1**2 > 5.5`, also at the default step sizes and `relTOL = 1e-9`; and the asymmetric path of an imperfect arch, whose extreme loads are those of their Eq. (15). The same arch with 40 BeamC elements is a test of `pyfe3d.solver`, `tests/test_solver_arc_length_fung_kaplan.py` of pyfe3d, whose arc-length methods reach the load factor 1.0 at the default tolerances only with the round-off floor of the convergence criteria, see above.

## 0.8.0 (2026-10-14)

### Performance: fewer evaluations and factorizations, same solutions

- `dR/dlbd` of a `calc_fext` that declares `lbd` is evaluated together with
  the tangent stiffness matrix, at the same iterate, instead of before every
  convergence check. The last one evaluated is the reference
  `lbd*dR/dlbd` of the relative criterion until the next update of the
  tangent, also across steps (the one of a failed step is not kept).
  Newton-Raphson no longer evaluates it at the converged iterate of each
  step, where the tangent is not needed, and with `modified_NR=True` it is
  evaluated once per update of the tangent instead of once per iteration.
  The arc-length methods keep the one of the converged state with its
  tangent when a step is repeated, and reuse the one of the origin in the
  first step. `calc_fext(c=c, lbd=lbd)` is always called right after
  `calc_kC` and `calc_kG` at the same configuration and load factor, such
  that the callables can share their computations. When `dR/dlbd`
  requires the tangent stiffness matrix, e.g. `[K_T] {c_p1}`, this removes
  one assembly per increment with full Newton-Raphson and most of them with
  modified Newton-Raphson. The convergence histories change slightly, the
  reference norm being lagged by at most one update of the tangent.
- Newton-Raphson reuses the LU factorization of the tangent stiffness
  matrix while it is not updated (`modified_NR=True`), as the arc-length
  methods already did; a singular tangent stops the iterations of the step
  with a warning, instead of a non-finite residual. `_factorize` moved
  from `arc_length` to `static`.
- The load correction iterations of the linear static analysis with a
  configuration-dependent load, `Analysis.static(NLgeom=False)` with a
  `calc_fext` that declares `c`, factorize `K0` once.
- `lb`: the estimate of the Cayley shift (power iterations, with one
  factorization of `K`) and the positive definiteness of `K` (one
  factorization) are computed once for all the attempts, the retry with a
  10 times larger shift and the dense fallback. New
  `linear_buckling._estimate_mu_max`; `_estimate_sigma` and `_eigsh_cayley`
  accept a precomputed `mu_max`, and `_check_eigenpairs` a precomputed
  `K_positive_definite`, a boolean or a function.

### Tests

- `tests/test_reuse.py`: number of calls to `calc_fext`, `calc_kC` and of
  factorizations in Newton-Raphson (full and modified) and in the arc-length
  methods, and the reuse in `lb` and in the linear static analysis with a
  configuration-dependent load.

## 0.7.0 (2026-10-13)

### New: load vectors that depend on the load factor

For prescribed displacements imposed exactly as linear constraints, e.g. by
the null-space method of `panels`, `c = Z c_r + c_p0 + lbd*c_p1`, the
derivative of the reduced residual with respect to the load factor,
`dR_r/dlbd = Z^T (F_ext - K_T(c) c_p1)`, depends on the configuration and on
the load factor. All the changes apply only when `calc_fext` declares the
new keyword argument `lbd` (a `**kwargs` does not count); the callables that
do not declare it are called exactly as before and give the same results.

- `callbacks.calc_load_vector(an, c, fext, lbd=None)` passes `lbd` to
  `calc_fext` when it declares it, together with `c` when it declares `c`.
  New `callbacks.is_load_factor_dependent(an)`. `inc` keeps its meaning, the
  factor that multiplies the returned vector, and the solvers keep calling
  `calc_fext(inc=1., ...)`.
- Arc-length methods (Riks and Crisfield): the tangent predictor and the
  bordered solutions use `calc_fext(inc=1., c=c, lbd=lbd)` at the current
  configuration and load factor, and the displacement scale `u_ref` is the
  linear solution of `dR/dlbd` at `c = 0`, `lbd = 0`, since the reference
  vector `calc_fext(inc=1.)` is null under a prescribed displacement alone.
- Newton-Raphson: the predictor of the first increment is the linear
  solution of `dR/dlbd` at `c = 0`, `lbd = 0`, instead of the linear
  solution of the null reference vector; the next increments keep the
  extrapolation of the previous one.
- Relative convergence criterion: `lbd*dR/dlbd`, evaluated at the current
  iterate before each convergence check, is an additional reference, new
  keyword `q_ref` of `newton_raphson._check_convergence`; it remains
  meaningful when `dR/dlbd` vanishes at the origin, e.g. a prescribed
  displacement whose coupling with the free degrees of freedom starts
  quadratically. The arc-length methods reuse it in the bordered solution
  at the same iterate; with `modified_NR=True` it costs one call to
  `calc_fext` per iteration. Under a prescribed displacement
  alone, the reference vector and the reduced internal force vector both
  vanish at equilibrium and the criterion could not be satisfied without
  `absTOL`.
- The linear static analysis, `Analysis.static(NLgeom=False)`, never passes
  `lbd`.
- Tests: `tests/test_load_factor_dependent_load.py`, three hardening springs
  in series with the displacement of the last node prescribed, reduced to two
  degrees of freedom, compared with the exact solution, including a constant
  and an incremented prescribed displacement with a load, and the quadratic
  convergence of the Riks method, and `dR/dlbd = 0` at the origin. Docs: `ex_prescribed_displacements.rst`,
  `callbacks` and `Analysis`.

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

### Continuous integration

- The GitHub Actions run on Node.js 24, whose Node.js 20 versions are
  deprecated: `actions/checkout` v7, `actions/setup-python` v7,
  `actions/setup-node` v7, `codecov/codecov-action` v7 and
  `softprops/action-gh-release` v3.
- `fail-fast: false` in the matrix of `pytest.yml`, such that a failed job
  does not cancel the others.

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
  `ncv`. The eigenvectors of all the passes are combined with a
  Rayleigh-Ritz projection, solved with the LAPACK driver `dsyevr`, which
  does not use `dsteqr`: the eigenvectors of different passes were only
  orthogonal to the accuracy of ARPACK, e.g. `v.T @ K @ v` deviated from
  the identity by `1.1e-6` for 25 eigenpairs of a cylinder of panels on
  Linux, now `1e-10`, and the eigenvalues are about 10 times more
  accurate. The cap can be overridden with `arpackutils.ARPACK_MAX_NCV` or the
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
