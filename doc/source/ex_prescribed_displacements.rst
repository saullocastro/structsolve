Prescribed displacements imposed exactly
========================================

A prescribed displacement can be imposed exactly as a linear constraint on
the degrees of freedom, `\boldsymbol{c} = \boldsymbol{Z} \boldsymbol{c}_r + \boldsymbol{c}_{p0} + \lambda
\boldsymbol{c}_{p1}`, where `\boldsymbol{Z}` is a basis of the null space of the constraints and
`\boldsymbol{c}_{p0}`, `\boldsymbol{c}_{p1}` are particular solutions for the constant and the
incremented prescribed values, as done by the null-space method of
`panels <https://github.com/saullocastro/panels>`_. The reduced residual is

.. math::

    \boldsymbol{R}_r(\boldsymbol{c}_r, \lambda) = \boldsymbol{Z}^\top \left(
    \boldsymbol{F}_{int}(\boldsymbol{Z} \boldsymbol{c}_r + \boldsymbol{c}_{p0} + \lambda \boldsymbol{c}_{p1}) -
    \lambda \boldsymbol{F}_{ext}\right)

and minus its derivative with respect to the load factor,

.. math::

    -\frac{\partial \boldsymbol{R}_r}{\partial \lambda} = \boldsymbol{Z}^\top \left(\boldsymbol{F}_{ext} -
    \boldsymbol{K}_T(\boldsymbol{c}) \boldsymbol{c}_{p1}\right)

depends on the configuration and on the load factor. ``calc_fext`` receives
both when it declares the keyword arguments ``c`` and ``lbd``, see
:mod:`.callbacks`. Under a prescribed displacement alone, `F_{ext} = 0`, the
reference vector ``calc_fext(inc=1.)`` and the reduced internal force vector
both vanish at equilibrium, and the solvers use `-\partial \boldsymbol{R}/\partial
\lambda` for the displacement scale of the arc-length methods, the predictor
of the first Newton-Raphson increment and the convergence
criterion.

Three hardening springs in series, with the displacement of the last node
prescribed:

.. literalinclude:: ../../tests/test_load_factor_dependent_load.py
    :pyobject: SpringsProblem

.. literalinclude:: ../../tests/test_load_factor_dependent_load.py
    :pyobject: test_displacement_control_relative_criterion

A constant prescribed displacement together with an incremented one and a
load:

.. literalinclude:: ../../tests/test_load_factor_dependent_load.py
    :pyobject: test_constant_and_incremented_prescriptions

Both together: the null-space method and ``constraint_values``
--------------------------------------------------------------

A displacement prescribed by the null-space method, through ``calc_fext``
with ``c`` and ``lbd``, combines with the non-homogeneous constraints
`\boldsymbol{C}^\top \boldsymbol{c} = \lambda \boldsymbol{d}` of
``Analysis.constraints`` and ``Analysis.constraint_values`` on the reduced
problem: the solutions of `-\partial \boldsymbol{R}/\partial \lambda` have
the border `\boldsymbol{d}`, those of the residual a null border. The
springs above, with a load on the node 2 and either the node 1 or the
relative displacement of the nodes 1 and 2 prescribed as well, against
the problem with both prescriptions eliminated by hand, a single unknown
found by :func:`scipy.optimize.brentq`, for all the non-linear solvers:

.. literalinclude:: ../../tests/test_constraint_values_load_factor.py
    :pyobject: reference

.. literalinclude:: ../../tests/test_constraint_values_load_factor.py
    :pyobject: test_path
