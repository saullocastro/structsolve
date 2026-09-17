Newton-Raphson solver (:mod:`structsolve.newton_raphson`)
=========================================================

.. automodule:: structsolve.newton_raphson

The solver is used through :meth:`.Analysis.static` with
``Analysis.NL_method = 'NR'``.

.. autofunction:: structsolve.newton_raphson._solver_NR

Convergence and divergence checks
---------------------------------

.. autofunction:: structsolve.newton_raphson._check_convergence

.. autofunction:: structsolve.newton_raphson._check_divergence

.. autofunction:: structsolve.newton_raphson._NR_iterations

Constants
---------

.. autodata:: structsolve.newton_raphson.NUM_ITER_BEFORE_DIVERGENCE_CHECK

.. autodata:: structsolve.newton_raphson.DIVERGENCE_FACTOR

.. autodata:: structsolve.newton_raphson.TOO_SLOW_WINDOW

.. autodata:: structsolve.newton_raphson.LINE_SEARCH_ALPHA

.. autodata:: structsolve.newton_raphson.INC_GROWTH_FACTOR

.. autodata:: structsolve.newton_raphson.INC_CUT_FACTOR
