Non-linear static analysis with the arc-length methods
======================================================

The arc-length methods, described in :func:`.arc_length._solver_arc_length`,
trace equilibrium paths that include limit points, where the Newton-Raphson
method with load control fails. The shallow von Mises truss below, loaded
through a linear spring, softens until a limit point, snaps through to an
inverted configuration, and stiffens again:

.. literalinclude:: ../../tests/analytic_problems.py
    :pyobject: von_mises_truss

The callables are defined as in the example
:doc:`ex_newton_raphson`. With a load of 1.5 times the limit load, the
arc-length methods, selected with ``NL_method='arc_length_riks'`` or
``NL_method='arc_length_crisfield'``, trace the path through the limit point,
where the load factor decreases and becomes negative, until the load factor
reaches exactly 1.0 on the inverted configuration. The code is extracted from
one of the ``structsolve`` unit tests:

.. literalinclude:: ../../tests/test_arc_length.py
    :pyobject: test_arc_length_snap_through
