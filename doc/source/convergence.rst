Convergence and troubleshooting of non-linear analyses
======================================================

The non-linear analyses of :class:`.Analysis` solve the equilibrium
`\{R\} = \lambda \{F_{ext}\} - \{F_{int}(c)\} = \{0\}` in increments of the
load factor (Newton-Raphson) or of the arc length (Riks and Crisfield). The
attributes of :class:`.Analysis` listed below control the iterations of each
increment and the size of the increments.


Convergence criteria
--------------------

An iteration converged when either criterion is satisfied, see
:func:`.newton_raphson._check_convergence`:

- ``relTOL``, default ``1e-6``: the Euclidean norm of `\{R\}` is not larger
  than ``relTOL`` times the largest norm of the force vectors of the
  increment, `\lambda\{F_{ext}\}` and `\{F_{int}\}`, and of `\lambda
  \partial R/\partial\lambda` for load vectors that depend on the load
  factor, see :mod:`.callbacks`. It does not depend on the units of the
  model as a whole.
- ``absTOL``, default ``None``: the largest absolute value of `\{R\}` is
  not larger than ``absTOL``, a force in the units of the model.

``None`` disables a criterion, and at least one must be given. A tolerance
below the round-off error of `\{R\}` cannot be met: every step then fails.


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

The relative criterion gives the same increments and solutions when the
forces and the displacements of a model are scaled:

.. literalinclude:: ../../tests/test_arc_length.py
    :pyobject: test_units_independence
