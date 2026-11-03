Follower loads and unsymmetric matrices
=======================================

Loads that depend on the configuration, such as follower (hydrostatic)
pressures, are supported through optional keyword arguments of the callables
of :class:`.Analysis`, see :mod:`.callbacks`: ``calc_fint``, ``calc_kC`` and
``calc_kG`` receive the load factor ``inc`` when they declare it, and
``calc_fext`` receives the current configuration ``c``, returning the load
vector `-\partial \boldsymbol{R}/\partial \lambda` used by the arc-length methods. Their
load stiffness makes the tangent stiffness matrix unsymmetric in general,
which the solvers support: :func:`.solve` and the non-linear solvers use LU
factorizations, :func:`.lb` selects the solvers of unsymmetric matrices when
needed, ``symmetric=None``, and :func:`.freq` uses general eigenvalue solvers
by default.

Non-linear analyses
-------------------

Hardening springs under a load `\lambda(\boldsymbol{q}_0 - \boldsymbol{K}_{F_{ref}} \boldsymbol{c})` with an
unsymmetric load stiffness matrix `\boldsymbol{K}_F = -\partial \boldsymbol{F}/\partial \boldsymbol{c} = \lambda
\boldsymbol{K}_{F_{ref}}`: the callables receive the load factor and the
configuration, and Newton-Raphson and the arc-length methods converge to the
equilibrium path. The Riks method converges quadratically only with the load
vector of the current configuration:

.. literalinclude:: ../../tests/test_follower_loads.py
    :pyobject: FollowerProblem

.. literalinclude:: ../../tests/test_follower_loads.py
    :pyobject: test_riks_quadratic_only_with_configuration_load_vector

Inertia of unsymmetric tangent stiffness matrices
-------------------------------------------------

The symmetric indefinite factorization of PARDISO counts the negative
eigenvalues of a symmetric tangent stiffness matrix (``Analysis.inertia``
with ``linear_solver='pardiso'``); an unsymmetric one is factorized by
SuperLU, whose LU factorization gives only the sign of the determinant,
the parity of the number of negative real eigenvalues. With
``Analysis.inertia_num_eigvals = k``, the number of negative real
eigenvalues among the `k` of smallest magnitude is computed at every
converged increment by shift-invert at `\sigma = 0` with ARPACK, with the
LU factorization of the converged state, and recorded in
``Analysis.num_negative_eigvals``, such that ``max_inertia_jump`` applies,
see :func:`.newton_raphson._eigvals_inertia`.

It detects the divergence points, where a real eigenvalue crosses zero,
the static critical points of the path: limit points and bifurcations. It
does not detect flutter: a complex pair `a \pm i b` whose real part
becomes negative is not a static critical point, the tangent staying
regular, and it is not counted (the number of such pairs among the `k` is
logged), the kinetic criterion of :func:`.freq` being needed, see below.
Negative real eigenvalues beyond the `k` of smallest magnitude are missed;
an odd number of them is reported, since the parity of the count then
differs from the sign of the determinant.

A problem whose tangent is `\mathrm{diag}(k_0, \boldsymbol{S}
(\boldsymbol{D}_0 + \lambda \boldsymbol{D}_1) \boldsymbol{S}^{-1})`, with
two real eigenvalues crossing zero at `\lambda = 0.27` and `0.63` and a
complex pair at `0.47`: the count goes 0, 1, 2, the sign of the
determinant +1, -1, +1, and the count agrees with
:func:`numpy.linalg.eigvals` at every increment:

.. literalinclude:: ../../tests/test_inertia_unsymmetric.py
    :pyobject: test_count_along_path

Linear static analysis
----------------------

With a configuration-dependent load, ``Analysis.static(NLgeom=False)`` solves
the geometrically linear problem `\boldsymbol{K}_{C0} \boldsymbol{c} = \boldsymbol{F}(\boldsymbol{c})`:

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
