Eigenvalue solvers
==================

The eigenvalue analyses :func:`.lb` and :func:`.freq` rely on sparse solvers
and keep dense solvers only as a last resort. Sparse solvers scale with the
number of non-null entries of the matrices, while a dense solution scales
with the cube of the number of degrees of freedom. This matters in
particular in the browser (Pyodide), where dense LAPACK is single-threaded,
and for follower loads, whose load stiffness couples almost all the degrees
of freedom.


Solver policy
-------------

- ``sparse_solver=True``, the default, always runs the sparse solvers first,
  for every problem size, symmetric or unsymmetric. Every result is
  verified before it is returned: the relative residual of each eigenpair
  and, for symmetric matrices, the inertia of a shifted matrix, which tells
  whether an eigenvalue was missed (Sylvester's law of inertia).
- When all the sparse solvers failed (verification, ARPACK error or no
  convergence, singular factorization, no positive real load multiplier
  for unsymmetric matrices...), a dense solver is used as a last resort, if
  the dense problem has at most ``max_dense_size`` degrees of freedom
  (default ``2000``, ``0`` disables the dense fallback). A
  :class:`.DenseFallbackWarning` gives the reasons of the failures, so that
  a dense solution is never silent, and the dense result is verified as
  well.
- Otherwise a ``RuntimeError`` lists every attempted solver and the reason
  of its failure. No silently wrong result is returned.
- ``sparse_solver=False`` explicitly solves the full problem with a dense
  solver, :func:`scipy.linalg.eigh` or :func:`scipy.linalg.eig`, without
  trying the sparse solvers, returning all the eigenvalues. It is a
  reference for validations and for the callers that need all of them.

The dense fallback makes ``structsolve`` robust in environments where
ARPACK is unreliable, see below, while normal runs never pay for a dense
solution. To turn the fallback into an error, e.g. in a test suite::

    import warnings
    from structsolve import DenseFallbackWarning
    warnings.simplefilter('error', DenseFallbackWarning)


Linear buckling, symmetric matrices
-----------------------------------

:func:`scipy.sparse.linalg.eigsh` in Cayley mode, with a shift estimated by
power iterations on `[K]^{-1}[K_G]`, retried with a 10 times larger shift.
The dense fallback condenses out the degrees of freedom where `[K_G]` has
null rows and solves the condensed problem with :func:`scipy.linalg.eigh`.


Linear buckling, unsymmetric matrices
-------------------------------------

With follower loads many negative or complex load multipliers may have a
smaller modulus than the critical one, e.g. under an internal pressure, and
the multipliers of smallest modulus may contain no positive real one. The
sparse solver is a search of shifts `s` along the positive real axis: at
each shift, :func:`scipy.sparse.linalg.eigs` finds the load multipliers
nearest `s`, within a disc that contains all of them. The shifts move until
the lowest positive real multiplier found is inside the union of the discs
that covers the positive real axis from zero, such that none was missed. A
parity check of the sign of `\det([K] + s [K_G])` confirms that an even
number, normally zero, of real multipliers lies below the critical one. See
:func:`structsolve.linear_buckling._eigs_shift_search`.


Frequency analysis
------------------

:func:`scipy.sparse.linalg.eigs`, or :func:`scipy.sparse.linalg.eigsh` with
``symmetric=True``, in shift-invert mode with a negative shift. The dense
fallback returns the lowest frequencies, or the eigenvalues closest to the
shift for unsymmetric matrices.


Intel MKL and ARPACK
--------------------

The LAPACK routine ``dsteqr`` of Intel MKL 2024.2.0 to 2025.0.0 returns
wrong eigenvectors for matrices larger than 32 x 32. ARPACK's symmetric
drivers, used by :func:`scipy.sparse.linalg.eigsh`, apply it to a matrix of
size ``ncv``, the dimension of the Krylov subspace, and then return wrong
eigenpairs or raise ``ArpackError -8``, e.g. with some Anaconda builds of
SciPy. :mod:`structsolve.arpackutils` detects such a LAPACK and keeps
``ncv <= 32``, computing more than 15 eigenpairs in several passes with
deflation. The eigenvectors of all the passes are combined with a
Rayleigh-Ritz projection, solved with the LAPACK driver ``dsyevr``, which
does not use ``dsteqr``, such that they are `[M]`-orthonormal also across
the passes, where ARPACK alone gives them orthogonal only to its accuracy.
The detection can be overridden, e.g. to lift the cap with MKL 2025.0.1 or
newer when its version cannot be read::

    STRUCTSOLVE_ARPACK_MAX_NCV=0 python my_script.py

or, in Python::

    import structsolve.arpackutils
    structsolve.arpackutils.ARPACK_MAX_NCV = 0

The non-symmetric drivers of :func:`scipy.sparse.linalg.eigs` are not
affected.


In the browser
--------------

In Pyodide (``sys.platform == 'emscripten'``) SciPy uses OpenBLAS, ``ncv`` is
not capped, and :func:`structsolve.arpackutils.release_memory` collects the
reference cycles of SciPy's ARPACK wrappers after each call, which otherwise
keep the sparse factorizations alive until the WebAssembly heap is
exhausted. The dense solvers of unsymmetric matrices call
:func:`scipy.linalg.eig` with complex matrices, since the real drivers crash
the Python process for some matrices with Intel MKL 2025.0.0.
