Follower loads and unsymmetric matrices
=======================================

Loads that depend on the configuration, such as follower (hydrostatic)
pressures, are supported through optional keyword arguments of the callables
of :class:`.Analysis`, see :mod:`.callbacks`: ``calc_fint``, ``calc_kC`` and
``calc_kG`` receive the load factor ``inc`` when they declare it, and
``calc_fext`` receives the current configuration ``c``, returning the load
vector `\partial R/\partial \lambda` used by the arc-length methods. Their
load stiffness makes the tangent stiffness matrix unsymmetric in general,
which the solvers support: :func:`.solve` and the non-linear solvers use LU
factorizations, :func:`.lb` selects the solvers of unsymmetric matrices when
needed, ``symmetric=None``, and :func:`.freq` uses general eigenvalue solvers
by default.

Non-linear analyses
-------------------

Hardening springs under a load `\lambda(\{q_0\} + [Q]\{c\})` with an
unsymmetric `[Q]`: the callables receive the load factor and the
configuration, and Newton-Raphson and the arc-length methods converge to the
equilibrium path. The Riks method converges quadratically only with the load
vector of the current configuration:

.. literalinclude:: ../../tests/test_follower_loads.py
    :pyobject: FollowerProblem

.. literalinclude:: ../../tests/test_follower_loads.py
    :pyobject: test_riks_quadratic_only_with_configuration_load_vector

Linear static analysis
----------------------

With a configuration-dependent load, ``Analysis.static(NLgeom=False)`` solves
the geometrically linear problem `[K_0]\{c\} = \{F(c)\}`:

.. literalinclude:: ../../tests/test_follower_freq_static.py
    :pyobject: test_analysis_linear_static_with_configuration_dependent_load

Linear buckling with unsymmetric matrices
-----------------------------------------

.. literalinclude:: ../../tests/test_follower_loads.py
    :pyobject: test_lb_unsymmetric_real

Under follower loads, many negative or complex load multipliers may have a
smaller modulus than the critical one, e.g. when the pressure is reversed.
The sparse solver of :func:`.lb` searches the lowest positive real load
multiplier along the positive real axis, see :doc:`eigensolvers`. The
cantilevers of Schweizerhof and Ramm (1984), with the matrices of
`panels <https://github.com/saullocastro/panels>`_, original and reversed:

.. literalinclude:: ../../tests/test_linear_buckling_unsymmetric.py
    :pyobject: test_lb_follower_shift_search

Flutter: the kinetic criterion
------------------------------

Ziegler's double pendulum under a follower force never diverges, its
stiffness matrix is never singular, and :func:`.lb` finds no critical load.
It loses stability by flutter at `P_f = (7/2 - \sqrt{2}) k/l`, where two
natural frequencies of :func:`.freq` coalesce and become complex:

.. literalinclude:: ../../tests/test_follower_freq_static.py
    :pyobject: ziegler

.. literalinclude:: ../../tests/test_follower_freq_static.py
    :pyobject: test_freq_ziegler_flutter

The follower pressure loads of `panels <https://github.com/saullocastro/panels>`_
use all these features, see ``Shell.add_pressure_load(..., follower=True)``.
