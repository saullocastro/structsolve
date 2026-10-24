Boundary conditions and vector sizes
====================================

The solvers of ``structsolve`` receive the matrices of the whole model and
impose no boundary condition themselves. A homogeneous boundary condition,
i.e. a degree of freedom (dof) constrained to zero, is given to them in one
of the three forms below, which lead to the same solution but to vectors of
different sizes.


Null rows and columns, removed by the solvers
---------------------------------------------

A constrained dof has a null row and a null column in the stiffness matrix,
e.g. when a Ritz or finite element model does not assemble its terms or
zeroes them. :func:`.solve`, :func:`.static`, :func:`.lb`, :func:`.freq` and
the non-linear solvers of :class:`.Analysis` remove these dofs with
:func:`.sparseutils.remove_null_cols` before solving and return vectors with
the size of the model, with zeros at the removed dofs:

- :func:`.solve` and :func:`.static`: ``x[i] = 0`` at the removed dofs; the
  terms of the right-hand side at these dofs are ignored;
- :func:`.lb` and :func:`.freq`: ``eigvecs`` has ``K.shape[0]`` rows and
  null rows at the removed dofs. The dofs are those of the first matrix,
  ``K``, and the same rows and columns are removed from ``KG`` or ``M``;
- :class:`.Analysis`: every factorization of the tangent stiffness matrix
  removes its null rows and columns, and ``an.cs`` contains vectors with the
  size of the model.

A dof is removed only when both its row and its column are null, such that
an unsymmetric matrix never loses an equation whose row is not null. Only
the stored values different from zero count: a very small value, e.g. of a
penalty stiffness, keeps the dof. With ``skip_null_cols=True``, :func:`.lb`
and :func:`.freq` skip the removal, which is only valid for non-singular
matrices.


Partition of the free dofs, by the user
---------------------------------------

The matrices may also be restricted to the free dofs `u` before calling the
solvers, e.g. ``K[free, :][:, free]``. The solvers then return vectors of
the size of the free dofs, expanded by the user with ``x_full[free] = x``.
The prestress that generates `[K_G]`, see :doc:`lb_prestress`, must be
computed from the expanded vector.


Null space of the constraints, reduce and expand
------------------------------------------------

General linear constraints `[C]\{c\} = \{0\}`, e.g. multi-point constraints
or the connection of several domains, are imposed with a basis `[T]` of the
null space of `[C]`, `\{c\} = [T]\{c_r\}`, which reduces the matrices to

.. math::

    [K_r] = [T]^T[K][T], \qquad [K_{G_r}] = [T]^T[K_G][T], \qquad
    [M_r] = [T]^T[M][T], \qquad \{f_r\} = [T]^T\{f\}

The solvers return the reduced vectors `\{c_r\}` and the eigenvectors in the
reduced coordinates, expanded by the user with `\{c\} = [T]\{c_r\}`. The
partition of the free dofs is the particular case where `[T]` contains the
columns of the identity matrix of the free dofs. Prescribed values different
from zero add a particular solution, `\{c\} = [T]\{c_r\} + \{c_p\}`, see
:doc:`ex_prescribed_displacements`.


Example
-------

The chain of hardening springs of ``tests/analytic_problems.py``, see
:doc:`ex_newton_raphson`, with its free end constrained to zero:

.. literalinclude:: ../../tests/analytic_problems.py
    :pyobject: hardening_springs

The three forms give the same static solution, natural frequencies and
modes, the first one with the size of the model:

.. literalinclude:: ../../tests/test_static.py
    :pyobject: test_boundary_conditions_three_ways
