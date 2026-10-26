Assembling matrices with sparseutils
====================================

The matrices passed to the solvers are SciPy sparse matrices. The module
:mod:`.sparseutils` helps assembling them from the contributions of the
elements, of the integration points or of the pairs of approximation
functions of a model.


Symmetric matrices from the upper triangle
------------------------------------------

The stiffness, geometric stiffness and mass matrices of conservative
problems are symmetric. Only their upper triangle, row `\le` column, needs
to be computed and stored, as three arrays of row indices, column indices
and values of a :class:`scipy.sparse.coo_matrix`, in which repeated entries
are summed when the matrix is converted to another format.
:func:`.sparseutils.make_symmetric` completes the lower triangle from the
upper one, ignoring the terms already below the diagonal, and
:func:`.sparseutils.finalize_symmetric_matrix` also checks that no stored
value is ``nan`` or ``inf`` and returns a CSR matrix, the format used by the
solvers.


Unsymmetric matrices
--------------------

A matrix with a skew-symmetric part, e.g. a load stiffness matrix, is the
sum of its symmetric part, `(\boldsymbol{A} + \boldsymbol{A}^\top)/2`, and of its skew-symmetric
part, `(\boldsymbol{A} - \boldsymbol{A}^\top)/2`, assembled from their upper triangles with
:func:`.sparseutils.make_symmetric` and
:func:`.sparseutils.make_skew_symmetric`. Unsymmetric matrices may also be
assembled in full. :func:`.linear_buckling.is_symmetric` and
:func:`.sparseutils.is_symmetric` check the symmetry of the result, with a
relative norm and entry by entry, respectively.


Constrained degrees of freedom
------------------------------

The rows and columns of the constrained dofs may be left null, and the
solvers remove them with :func:`.sparseutils.remove_null_cols`, see
:doc:`boundary_conditions`.


Example
-------

A chain of springs assembled from the upper triangles of its element
matrices, and a general matrix from its symmetric and skew-symmetric
parts:

.. literalinclude:: ../../tests/test_sparseutils.py
    :pyobject: test_assembly_upper_triangle
