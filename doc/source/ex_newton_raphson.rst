Non-linear static analysis with the Newton-Raphson method
=========================================================

A non-linear static analysis with :class:`.Analysis` requires four callables,
which calculate the external force vector, the internal force vector, the
constitutive stiffness matrix and the geometric stiffness matrix. The tangent
stiffness matrix `[K_T] = [K_C] + [K_G]` should be the exact derivative of the
internal force vector, which gives quadratic convergence to the Newton-Raphson
method, described in :func:`.newton_raphson._solver_NR`.

The class below, used in the ``structsolve`` unit tests, wraps analytic
problems into these callables:

.. literalinclude:: ../../tests/analytic_problems.py
    :pyobject: Problem

One of these problems is a chain of springs with a cubic hardening, where
`\{F_{int}\} = [K]\{u\} + a \{u\}^3`:

.. literalinclude:: ../../tests/analytic_problems.py
    :pyobject: hardening_springs

The full load is applied in a single step, starting from the linear solution,
and the order of convergence of the iterations is verified:

.. literalinclude:: ../../tests/test_newton_raphson.py
    :pyobject: test_nr_full_load_quadratic_convergence

The analysis can also be performed in increments, with ``initialInc`` defining
the size of the first load increment:

.. literalinclude:: ../../tests/test_newton_raphson.py
    :pyobject: test_nr_incremental_hardening
