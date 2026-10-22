Frequency analysis
==================

The function :func:`.freq` solves the eigenvalue problem
`([K] + \lambda^2 [M])\{u\} = \{0\}`, where `\lambda^2 = -\omega_n^2`
and `\omega_n` is the natural frequency in rad/s. The returned ``lambda2``
are therefore negative for a stable structure, and

.. code-block:: python

    omega = np.sqrt(-lambda2)   # rad/s
    f = omega/(2*np.pi)         # Hz

with ``np.sqrt(-lambda2 + 0j)`` when ``lambda2`` may be complex or positive,
e.g. beyond a flutter or a divergence load, see :doc:`ex_follower_loads`.
The code below is extracted from one of the ``structsolve`` unit tests:

.. literalinclude:: ../../tests/test_freq.py
    :pyobject: test_freq_simple


Options
-------

- ``symmetric``: ``False``, the default, uses the general solvers, valid
  for unsymmetric matrices, and returns complex arrays. ``True`` uses the
  solvers of symmetric matrices, which require a positive definite ``M``,
  and returns real arrays; ``None`` checks the symmetry of the matrices.
- Normalization: with ``symmetric=True``, the eigenvectors are normalized
  with respect to ``M``, `\{u_i\}^T[M]\{u_j\} = \delta_{ij}`. Otherwise
  the sparse solver returns eigenvectors normalized with respect to ``M``
  and the dense solver eigenvectors with unit Euclidean norm, see
  :func:`.freq`; an application that needs a given normalization applies it
  to the returned eigenvectors.
- ``sort``: ``sort=True``, the default, sorts ``lambda2`` and the
  eigenvectors together by increasing natural frequency and removes the
  eigenpairs with `\omega_n \le 10^{-6}` rad/s, e.g. rigid-body modes,
  such that fewer than ``num_eigvalues`` may be returned. ``sort=False``
  keeps the order of the solver.
- ``num_eigvalues``: the number of eigenvalues computed by the sparse
  solver, those closest to a negative shift, i.e. the lowest frequencies.
  See :doc:`eigen_accuracy` for the verification of the results.

The sorted eigenpairs, with and without a rigid-body mode:

.. literalinclude:: ../../tests/test_freq.py
    :pyobject: test_freq_sort


Prestressed structures
----------------------

The frequencies of a prestressed structure come from the stiffness matrix
of the prestressed state, e.g. ``K + lam*KG`` with the geometric stiffness
matrix ``KG`` of a reference load, see :doc:`lb_prestress`, or the tangent
stiffness matrix of a non-linear analysis. A compressive prestress reduces
the frequencies, and the lowest one vanishes at the buckling load.

For the transverse modes of the pinned column of ``tests/column.py``, with
lumped masses, the eigenvectors of ``K``, ``KG`` and ``M`` coincide and
`\omega_k^2(\lambda) = \omega_k^2 (1 - \lambda/\lambda_k)`, with
`\lambda_k` the load multipliers of :func:`.lb`:

.. literalinclude:: ../../tests/test_column.py
    :pyobject: test_freq_prestressed_column
