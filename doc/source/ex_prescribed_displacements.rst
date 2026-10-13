Prescribed displacements imposed exactly
========================================

A prescribed displacement can be imposed exactly as a linear constraint on
the degrees of freedom, `\{c\} = [T]\{c_r\} + \{c_{p0}\} + \lambda
\{c_{p1}\}`, where `[T]` is a basis of the null space of the constraints and
`\{c_{p0}\}`, `\{c_{p1}\}` are particular solutions for the constant and the
incremented prescribed values, as done by the null-space method of
`panels <https://github.com/saullocastro/panels>`_. The reduced residual is

.. math::

    \{R_r(c_r, \lambda)\} = [T]^T \left(\lambda \{F_{ext}\} -
    \{F_{int}([T]\{c_r\} + \{c_{p0}\} + \lambda \{c_{p1}\})\}\right)

and its derivative with respect to the load factor,

.. math::

    \frac{\partial R_r}{\partial \lambda} = [T]^T \left(\{F_{ext}\} -
    [K_T(c)]\{c_{p1}\}\right)

depends on the configuration and on the load factor. ``calc_fext`` receives
both when it declares the keyword arguments ``c`` and ``lbd``, see
:mod:`.callbacks`. Under a prescribed displacement alone, `F_{ext} = 0`, the
reference vector ``calc_fext(inc=1.)`` and the reduced internal force vector
both vanish at equilibrium, and the solvers use `\partial R/\partial
\lambda` for the displacement scale of the arc-length methods, the predictor
of the first Newton-Raphson increment and the relative convergence
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
