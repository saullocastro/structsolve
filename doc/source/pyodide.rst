Running in the browser (Pyodide)
================================

`Pyodide <https://pyodide.org>`_ is a distribution of Python compiled to
WebAssembly, which runs in a web browser or under Node.js
(``sys.platform == 'emscripten'``). ``structsolve`` is pure Python and only
requires NumPy and SciPy, which Pyodide distributes, so the wheel published
on PyPI is installed with ``micropip``::

    import micropip
    await micropip.install("structsolve")

after which every solver is used as in a native Python. The differences
come from the numerical libraries available in WebAssembly and from its
memory model.


Linear algebra in WebAssembly
-----------------------------

The SciPy of Pyodide is linked to OpenBLAS, not to Intel MKL. Hence:

- the cap of the number of Lanczos vectors ``ncv`` of ARPACK, needed with
  the faulty ``dsteqr`` of Intel MKL 2024.2.0 to 2025.0.0, is not applied:
  the functional probe :func:`.arpackutils.dsteqr_is_faulty` passes and the
  LAPACK is not MKL, see :func:`.arpackutils.arpack_max_ncv`, and
  :func:`scipy.sparse.linalg.eigsh` uses SciPy's default ``ncv``;
- the dense solvers of unsymmetric matrices call :func:`scipy.linalg.eig`
  with the real LAPACK drivers, whereas natively the matrices are made
  complex to avoid a crash of the real drivers of Intel MKL 2025.0.0;
- ``linear_solver='pardiso'`` is not available, since it calls the shared
  library ``mkl_rt`` of Intel oneMKL (:func:`.pardiso.load_mkl` raises
  ``ImportError``); the default ``linear_solver='splu'``, the SuperLU
  factorization of SciPy, is used.

Dense LAPACK runs on a single thread in the browser, which makes the policy
of the eigenvalue solvers, sparse solvers first and dense solvers only as a
last resort within ``max_dense_size``, more important than natively: a
sparse solution scales with the number of non-null entries of the matrices,
a dense one with the cube of the number of degrees of freedom.


Memory
------

The objects of :func:`scipy.sparse.linalg.eigsh` and
:func:`scipy.sparse.linalg.eigs` keep the SuperLU factorizations of the
shift-invert modes in reference cycles, which only the cyclic garbage
collector frees. In Pyodide the WebAssembly heap is exhausted before the
collector runs, e.g. after about ten ``eigs`` calls on a problem with 1,000
degrees of freedom, raising ``MemoryError``. :func:`.arpackutils.release_memory` is
therefore called after each ARPACK call: it runs :func:`gc.collect` in
Pyodide and does nothing natively.


Testing
-------

The test suite of the repository runs in Pyodide under Node.js: the wheel
built from the repository is installed with ``micropip``, and NumPy, SciPy
and pytest are those distributed with Pyodide, whose version is pinned in
``ci/pyodide/package.json``; the script is ``ci/pyodide/run_tests.mjs``.
The test that starts a subprocess to check the crash of the real LAPACK
drivers of Intel MKL is skipped in Pyodide.
