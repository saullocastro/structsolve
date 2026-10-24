Accuracy of the eigenvalue solutions
====================================

Every eigenpair returned by :func:`.lb` and :func:`.freq` passed the
verification described in :doc:`eigensolvers`: a relative residual not
larger than ``check_rtol`` and, for symmetric matrices, the inertia check,
which tells that no eigenvalue below the highest one returned is missing.
The parameters below control what is computed and how strictly it is
verified.


Repeated and clustered eigenvalues
----------------------------------

Structures with symmetries may have repeated eigenvalues, e.g. the two
equal critical load multipliers of the cylinder model of the example below.
The eigenvalue is then unique, but its eigenvectors are any basis of a
subspace: two solvers, or two runs with a different ``num_eigvalues``,
return the same eigenvalues and possibly different eigenvectors, all of
them valid. Eigenvectors of repeated or very close eigenvalues are compared
by their subspaces, e.g. with :func:`scipy.linalg.subspace_angles`, and not
one by one.


Number of eigenvalues
---------------------

``num_eigvalues`` eigenvalues are computed by the sparse solvers. The
inertia check guarantees that none is missing below the highest one
returned, but a group of repeated eigenvalues may be cut at the last
returned one, e.g. ``num_eigvalues=1`` returns one mode of a pair. A margin
above the number of modes needed avoids cutting a group whose modes are all
needed, e.g. for a modal superposition or a comparison of mode shapes.


Tolerances
----------

- ``check_rtol``: maximum relative residual of an eigenpair, default
  ``1e-3`` in :func:`.lb` and ``1e-6`` in :func:`.freq`. A smaller value
  rejects less accurate sparse solutions, at the cost of more dense
  fallbacks or a ``RuntimeError`` when ARPACK does not reach it.
- ``tol``: tolerance of ARPACK, ``0`` for the machine precision.


Dense solutions
---------------

- ``sparse_solver=False`` solves the full problem with a dense solver and
  returns all the eigenvalues. Its cost grows with the cube of the number
  of dofs; it is a reference for validations.
- ``max_dense_size``: largest problem solved with the dense fallback when
  the sparse solvers failed. ``0`` disables the fallback, such that a
  failure of the sparse solvers raises a ``RuntimeError``, and
  ``warnings.simplefilter('error', DenseFallbackWarning)`` turns the
  fallback into an error, see :doc:`eigensolvers`.


Examples
--------

Two equal critical load multipliers of a cylinder model of
`panels <https://github.com/saullocastro/panels>`_, whose matrices are saved
in ``tests/data``:

.. literalinclude:: ../../tests/test_linear_buckling.py
    :pyobject: test_lb_repeated_load_multipliers

The verification detects wrong eigenpairs and missing load multipliers:

.. literalinclude:: ../../tests/test_linear_buckling.py
    :pyobject: test_lb_check_eigenpairs

The sparse solver of :func:`.freq` against the dense solution:

.. literalinclude:: ../../tests/test_freq.py
    :pyobject: test_freq_lowest_frequencies
