How the documentation is organised
==================================

The documentation of ``structsolve`` is written in four cumulative levels,
each with its own place, so that a reader, or a tool, can go from what a
function does to why it does it without the levels repeating each other.

.. list-table::
    :header-rows: 1

    * - Level
      - Where
      - Content
    * - 1, functional
      - docstrings of the modules, classes, functions and methods, except
        their ``Notes`` and ``References`` sections; comments in the code;
        docstrings and comments of the tests; the ``README.md``
      - what each resource does: a one-line summary, an extended summary
        (what, not why), ``Parameters`` with type, unit, shape and
        default, ``Returns``, ``Raises`` and ``See Also``, and short
        ``Examples`` for the main entry points (:class:`.Analysis`,
        :meth:`.Analysis.static`, :func:`.solve`, :func:`.static`,
        :func:`.lb`, :func:`.freq`)
    * - 2, theory
      - the ``Notes`` and ``References`` sections of the docstrings, used
        only for theory
      - the theory that each resource implements, local to it: the
        equations in the notation of the code, the algorithm, its
        assumptions and limits, and the published sources
    * - 3, overview
      - the pages of the user guide of this documentation, e.g.
        :doc:`conventions`, :doc:`eigensolvers`, :doc:`convergence` and
        :doc:`pyodide`
      - how the pieces of level 2 fit together: the sign conventions shared
        by the solvers, the policy of the eigenvalue solvers, the
        convergence criteria, the boundary conditions, with links to the
        API instead of repeating level 2
    * - 4, worked examples
      - the notebooks in ``notebooks/`` at the root of the repository,
        rendered here as the worked examples and tutorials
      - complete analyses explained step by step, with their results
        compared with the literature and stored outputs

The API pages render levels 1 and 2 together. To find the theory of a
resource, read the ``Notes`` of its docstring; to find how a solver fits in
an analysis, read the pages of level 3.


The tests
---------

The tests in ``tests/`` are part of level 1, and their ``Notes`` and
``References`` of level 2. Each test states in its docstring what it
verifies and against what:

* a published value, table or equation of the literature, cited with its
  location, e.g. the frequencies of the clamped-free rod of Petyt (2010),
  Example 3.3, or the critical load of a simply supported plate of
  Timoshenko and Woinowsky-Krieger (1959), Eq. (227);
* a closed-form solution written in the test, with its source;
* a property that holds for any input, e.g. the agreement of the sparse
  and the dense eigenvalue solvers, the inertia of a shifted matrix, or
  the orthonormality of the eigenvectors;
* a regression of a failure found before, on matrices saved in
  ``tests/data`` from models of other libraries, with the model described
  in the docstring.

The modules of ``tests/`` that are not tests, e.g. ``ritz.py`` (the
Legendre-based functions of the Ritz models of the tests) and
``simo1990_utils.py``, are models shared by the tests and the notebooks.
