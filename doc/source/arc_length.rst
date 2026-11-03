Arc-length solvers (:mod:`structsolve.arc_length`)
==================================================

.. automodule:: structsolve.arc_length

.. autofunction:: structsolve.arc_length._solver_arc_length

Riks and Crisfield methods
--------------------------

The modules :mod:`structsolve.arc_length_riks` and
:mod:`structsolve.arc_length_crisfield` run the shared solver with the
constraint of each method.

.. autofunction:: structsolve.arc_length_riks._solver_arc_length_riks

.. autofunction:: structsolve.arc_length_crisfield._solver_arc_length_crisfield

Displacement control
--------------------

The indirect displacement control of :mod:`structsolve.displacement_control`
and the generalized displacement control of
:mod:`structsolve.generalized_displacement_control` run the same shared
solver, :func:`structsolve.arc_length._solver_arc_length`, with their own
constraint.

.. autofunction:: structsolve.displacement_control._solver_displacement_control

.. autofunction:: structsolve.generalized_displacement_control._solver_generalized_displacement_control

Critical points
---------------

.. autofunction:: structsolve.arc_length._critical_eigenvector

.. autofunction:: structsolve.arc_length._eigs_near_zero

Constants
---------

.. autodata:: structsolve.arc_length.PSI

.. autodata:: structsolve.arc_length.DESIRED_NUM_ITER

.. autodata:: structsolve.arc_length.MAX_GROWTH_FACTOR

.. autodata:: structsolve.arc_length.MIN_GROWTH_FACTOR

.. autodata:: structsolve.arc_length.MAX_NUM_STEPS

.. autodata:: structsolve.arc_length.BRANCH_SWITCH_MIN_COS

.. autodata:: structsolve.arc_length.BRANCH_SWITCH_MAX_TRIES

.. autodata:: structsolve.arc_length.EIGVEC_NUM_GUARD

.. autodata:: structsolve.arc_length.EIGVEC_MAX_ITER

.. autodata:: structsolve.arc_length.EIGVEC_TOL

.. autodata:: structsolve.arc_length.EIGVEC_MAX_SIZE
