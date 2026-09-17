Linear static analysis
======================

The function :func:`.solve` solves a linear system of equations, removing
the null rows and columns of the stiffness matrix, which typically correspond
to constrained degrees of freedom. The values of the solution at these
degrees of freedom are zero. The code below is extracted from one of the
``structsolve`` unit tests:

.. literalinclude:: ../../tests/test_static.py
    :pyobject: test_solve_with_null_columns

Linear static analyses can also be run with :func:`.static`, or with the class
:class:`.Analysis`, which receives callables that calculate the external force
vector and the stiffness matrix:

.. literalinclude:: ../../tests/test_analysis.py
    :pyobject: _make_linear_system

.. literalinclude:: ../../tests/test_analysis.py
    :pyobject: test_analysis_linear_static
