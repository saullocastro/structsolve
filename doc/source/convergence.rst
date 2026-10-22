Convergence and troubleshooting of non-linear analyses
======================================================

The non-linear analyses of :class:`.Analysis` solve the equilibrium
`\{R\} = \lambda \{F_{ext}\} - \{F_{int}(c)\} = \{0\}` in increments of the
load factor (Newton-Raphson) or of the arc length (Riks and Crisfield). The
attributes of :class:`.Analysis` listed below control the iterations of each
increment and the size of the increments.


Convergence criteria
--------------------

The criterion is selected with ``Analysis.convergence``, see
:mod:`.convergence` and :func:`.newton_raphson._check_convergence`:

- ``'crisfield'``, the default: Crisfield's criterion. Each component of a
  force-like vector is scaled with the diagonal of the linear stiffness
  matrix ``calc_kC()``, `D_j = |{K_0}_{jj}|`, over the free degrees of
  freedom, i.e. those whose row or column of ``calc_kC()`` is not null,

  .. math::

      \|v\|_D = \sqrt{\sum_j v_j^2/D_j}, \qquad
      \zeta = \frac{\|R\|_D}{\max\left(\|\lambda F_{ext}\|_D,
      \|F_{int}\|_D, \|\Delta\lambda F_{ext}\|_D,
      \|\lambda \, \partial R/\partial \lambda\|_D\right)}

  and the iteration converged when `\zeta \le` ``relTOL``, default
  ``1e-9``. `\Delta\lambda F_{ext}` is a reference of the arc-length
  methods only, and `\lambda \, \partial R/\partial \lambda` of load
  vectors that depend on the load factor, see :mod:`.callbacks`. The
  criterion does not depend on the units of each degree of freedom, e.g.
  translations and rotations, and the residual at the constrained degrees
  of freedom, e.g. a support reaction, does not enter it.
- ``'relative'``: the Euclidean norm of `\{R\}` is not larger than
  ``relTOL`` times the largest Euclidean norm of the same references, over
  all the degrees of freedom, the only criterion up to version 0.8.0, with
  the default ``relTOL=1e-6`` then. It does not depend on the units of the
  model as a whole, but it adds the squares of forces and of moments.

With either criterion, ``absTOL``, default ``None``, also accepts an
iteration whose largest absolute value of `\{R\}` is not larger than
``absTOL``, a force in the units of the model. ``None`` disables a
tolerance, and at least one must be given. A tolerance below the round-off
error of `\{R\}` cannot be met: every step then fails. The divergence and
too-slow checks and the line search use the norm of the selected criterion.


Increments and iterations
-------------------------

- ``initialInc``: first load increment; with the arc-length methods, the
  first arc length, corresponding to a load factor increment of
  ``initialInc`` along the initial tangent.
- ``maxInc`` and ``minInc``: largest and smallest increment. The analysis
  stops when a step fails with an increment smaller than ``minInc``.
- ``maxNumIter``: corrections per step before the step is repeated with a
  smaller increment; ``too_slow_TOL`` and the module constants of
  :mod:`.newton_raphson` detect diverging and too slow iterations.
- ``maxArcLength``: cumulative arc length after which the arc-length
  methods stop, with ``arc_length.MAX_NUM_STEPS`` converged steps at most.
- ``modified_NR``, ``compute_every_n``, ``kT_initial_state`` and
  ``line_search``: variants of the iterations, see :class:`.Analysis`.


Detecting a run that did not finish
-----------------------------------

The analyses finish at the load factor exactly ``1.0`` unless they stop
earlier, and ``an.increments`` and ``an.cs`` keep only the converged
increments. A run that did not finish is detected with::

    an.static(NLgeom=True)
    if not an.increments or an.increments[-1] != 1.:
        ...  # stopped at an.increments[-1], or no increment converged

The log messages of the solvers give the reason, e.g. ``WARNING: Minimum
step size ... achieved!``; they are not printed with ``silent=True``, see
:doc:`conventions`. A finished run is not necessarily a continuous path:
beyond a limit point, the load-controlled Newton-Raphson method may converge
to another branch of the equilibrium path, which a jump of the solution
between increments reveals. The arc-length methods trace the path through
the limit point.


Checking the tangent stiffness matrix
-------------------------------------

The tangent stiffness matrix ``calc_kC + calc_kG`` must be the derivative
of ``calc_fint``, `[K_T] = \partial F_{int}/\partial c`. An inconsistent
tangent stiffness matrix, e.g. a missing geometric stiffness matrix, does
not change the converged solution, which is controlled by the residual, but
the convergence becomes linear and may fail, such that the steps are cut
and the analysis may stop. Two checks:

- the order of convergence of the iterations, quadratic with an exact
  tangent stiffness matrix and full Newton-Raphson, in the log messages of
  the iterations;
- central finite differences of ``calc_fint`` compared with ``calc_kC +
  calc_kG`` at a few configurations.


Examples
--------

The analytic problems of ``tests/analytic_problems.py``, see
:doc:`ex_newton_raphson`. A finite-difference check of the tangent
stiffness matrix:

.. literalinclude:: ../../tests/test_newton_raphson.py
    :pyobject: tangent_error

.. literalinclude:: ../../tests/test_newton_raphson.py
    :pyobject: test_tangent_consistency_finite_differences

An inconsistent tangent stiffness matrix:

.. literalinclude:: ../../tests/test_newton_raphson.py
    :pyobject: test_inconsistent_tangent_converges_linearly

A tolerance that cannot be met, and a run stopped by ``maxArcLength``:

.. literalinclude:: ../../tests/test_newton_raphson.py
    :pyobject: test_nr_unreachable_tolerance_stops_without_increments

.. literalinclude:: ../../tests/test_arc_length.py
    :pyobject: test_arc_length_maxArcLength

Load control beyond a limit point:

.. literalinclude:: ../../tests/test_newton_raphson.py
    :pyobject: test_nr_jumps_over_limit_point

Both criteria give the same increments and solutions when the forces and
the displacements of a model are scaled:

.. literalinclude:: ../../tests/test_arc_length.py
    :pyobject: test_units_independence

Only Crisfield's criterion is independent of the units of each degree of
freedom, and only it ignores the residual of a support:

.. literalinclude:: ../../tests/test_arc_length.py
    :pyobject: rescaled_dof_problem

.. literalinclude:: ../../tests/test_arc_length.py
    :pyobject: test_crisfield_criterion_units_invariance

.. literalinclude:: ../../tests/test_arc_length.py
    :pyobject: test_crisfield_criterion_free_dofs_only
