Linear solvers of the static analyses
=====================================

Every iteration of a non-linear analysis of :class:`.Analysis` solves a
linear system with the tangent stiffness matrix, and the linear analysis
does so with the linear stiffness matrix when ``Analysis.constraints`` is
given. The factorization is chosen with ``Analysis.linear_solver``, see
``static.LINEAR_SOLVERS`` and :func:`.static._factorize`, whose Notes give
the algorithms. All of them remove the null rows and columns first, see
:doc:`boundary_conditions`, and border the matrix with the constraints of
``Analysis.constraints``.

.. list-table::
    :header-rows: 1

    * - ``linear_solver``
      - Factorization
      - Matrices
      - Inertia (``Analysis.inertia``)
    * - ``'splu'`` (default)
      - sparse LU of SuperLU
      - any
      - sign of the determinant; negative eigenvalues with
        ``Analysis.inertia_num_eigvals``
    * - ``'pardiso'``
      - sparse symmetric indefinite `\boldsymbol{L} \boldsymbol{D}
        \boldsymbol{L}^\top` of PARDISO (oneMKL)
      - symmetric; ``'splu'`` is used for an unsymmetric matrix
      - number of negative eigenvalues and sign of the determinant
    * - ``'pardiso_lu'``
      - sparse unsymmetric LU of PARDISO (oneMKL)
      - any
      - not available: ``inertia=True`` raises a :class:`ValueError`, and
        so do the options that require it, ``branch_switching``,
        ``max_inertia_jump`` and ``inertia_num_eigvals``
    * - ``'dense_lu'``
      - dense LU of LAPACK (``getrf``)
      - any
      - as ``'splu'``

The choice changes the speed, not the result: the solutions agree to the
round-off, such that an analysis takes the same increments and Newton-Raphson
iterations with any of them, unless a matrix is so ill-conditioned that the
round-off matters.


Choosing a solver for speed
---------------------------

- ``'splu'`` needs nothing beyond SciPy and works everywhere, also in the
  browser (:doc:`pyodide`). SuperLU recomputes its ordering at every
  factorization and runs on one thread.
- ``'pardiso'`` is the fastest for large symmetric tangents, e.g. the
  geometrically non-linear Ritz models of plates and shells under
  conservative loads, and it also gives the inertia.
- ``'pardiso_lu'`` is the fastest for large unsymmetric tangents, or
  symmetric ones whose inertia is not needed, e.g. the tangents of
  cohesive-zone models of delamination: PARDISO runs on several threads and
  keeps the analysis (ordering and symbolic factorization) of the sparsity
  pattern along the analysis, such that each new factorization is only
  numerical.
- ``'dense_lu'`` is the fastest for small matrices, and for matrices so
  dense that a sparse factorization gains nothing, e.g. the Ritz models
  with global functions, whose tangent has most of its terms; its cost
  grows with the cube of the size, whatever the sparsity, and it needs the
  memory of the dense matrix.

``'pardiso'`` and ``'pardiso_lu'`` require the shared library ``mkl_rt`` of
Intel oneMKL, see :mod:`.pardiso`; without it they raise an
:class:`ImportError` (``pip install mkl`` or ``structsolve[pardiso]``).
The eigenvalue analyses :func:`.lb` and :func:`.freq` accept only
``'splu'`` and ``'pardiso'``, see :func:`.lb`.


Accuracy checks and fallback of ``'pardiso_lu'``
------------------------------------------------

PARDISO pivots only within blocks of the matrix and perturbs the pivots that
are too small, which keeps the factorization fast and parallel but may make
it inaccurate for an ill-conditioned matrix. Hence ``'pardiso_lu'`` checks
the relative residual of every solution against
``static.PARDISO_LU_RTOL``; a factorization with perturbed pivots that fails
the check is made again with the parameters
``static.PARDISO_LU_FALLBACK_IPARM``, and when that fails as well, or when a
later solution fails the check, SuperLU factorizes the matrix and its
solution is used. Each fallback to SuperLU issues a :class:`RuntimeWarning`
and a log message; along an analysis, the next factorizations of the same
sparsity pattern go to the fallback directly. The analysis is then correct,
at the speed of ``'splu'`` or a little slower, see
:func:`.static._factorize`.

The accuracy of PARDISO depends on the version of oneMKL: on the tangent of
a cohesive-zone model, oneMKL 2025.0 perturbed tens of pivots and gave
wrong solutions with its default parameters, where oneMKL 2023.1 was
accurate, and the fallback parameters recovered solutions as accurate as
those of SuperLU. When the warnings of the fallback to SuperLU appear at
every iteration, ``'splu'`` or ``'dense_lu'`` is faster than
``'pardiso_lu'``.
