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
