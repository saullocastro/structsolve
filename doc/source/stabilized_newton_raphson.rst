Stabilized Newton-Raphson solver (:mod:`structsolve.stabilized_newton_raphson`)
===============================================================================

.. automodule:: structsolve.stabilized_newton_raphson

The solver is used through :meth:`.Analysis.static` with
``Analysis.NL_method = 'stabilized_NR'``, the viscous forces being scaled
by ``Analysis.stabilization_factor``. The status of each accepted state is
in ``Analysis.stabilization_status`` and ``Analysis.damped_residuals``.

.. autofunction:: structsolve.stabilized_newton_raphson._solver_stabilized_NR

Stages of an increment
----------------------

.. autofunction:: structsolve.newton_raphson._stabilized_iterations
