Notation and sign conventions
=============================

Static analyses
---------------

`\boldsymbol{c}` is the vector of the degrees of freedom of the model, the solution
vectors of ``an.cs``, and `\lambda` the load factor, the values of
``an.increments``. The residual force vector of the non-linear analyses is

.. math::

    \boldsymbol{R}(\boldsymbol{c}, \lambda) = \lambda \boldsymbol{F}_{ext} - \boldsymbol{F}_{int}(\boldsymbol{c})

with `\boldsymbol{F}_{ext}` = ``calc_fext(inc=1.)`` and `\boldsymbol{F}_{int}` =
``calc_fint(c)``, and the tangent stiffness matrix is

.. math::

    \boldsymbol{K}_T = \boldsymbol{K}_C + \boldsymbol{K}_G = \frac{\partial \boldsymbol{F}_{int}}{\partial \boldsymbol{c}}
          = -\frac{\partial \boldsymbol{R}}{\partial \boldsymbol{c}}

with `\boldsymbol{K}_C` = ``calc_kC(c, NLgeom=True)`` and `\boldsymbol{K}_G` = ``calc_kG(c,
NLgeom=True)``. A configuration-dependent load `\boldsymbol{F}(\boldsymbol{c})` has the load
stiffness matrix `-\partial \boldsymbol{F}/\partial \boldsymbol{c}`, which enters `\boldsymbol{K}_T`, see
:mod:`.callbacks`. The linear static analysis solves `\boldsymbol{K}_{C0} \boldsymbol{c} =
\boldsymbol{F}_{ext}`, with `\boldsymbol{K}_{C0}` = ``calc_kC()``.


Eigenvalue analyses
-------------------

.. list-table::
    :header-rows: 1

    * - Analysis
      - Eigenvalue problem
      - Returned eigenvalues
    * - :func:`.lb`
      - `(\boldsymbol{K} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}`
      - load multipliers `\lambda`, the positive ones first, in increasing
        order
    * - :func:`.freq`
      - `(\boldsymbol{K} + \lambda^2 \boldsymbol{M}) \boldsymbol{u} = \boldsymbol{0}`
      - ``lambda2`` `= -\omega_n^2`, negative for a stable structure

For :func:`.lb`, `\boldsymbol{K}_G` is the geometric stiffness matrix of the
reference load, the critical load being `\lambda_1` times the reference
load, and a negative load multiplier corresponds to the reversed reference
load, see :doc:`lb_prestress`. For :func:`.freq`, `\omega_n` is in rad/s, see
:doc:`ex_frequency`; complex ``lambda2`` appear beyond a flutter load, and
a positive real ``lambda2`` indicates divergence, see
:doc:`ex_follower_loads`. The eigenvectors are the columns of ``eigvecs``,
with the size of the model, see :doc:`boundary_conditions`.


Units
-----

``structsolve`` has no units: the matrices and vectors of a model must use
a consistent system of units, the eigenvalues of :func:`.freq` are in
rad\ :sup:`2`/s\ :sup:`2` when the stiffness and the mass matrices are in
consistent units of force, length and time, and the arc length of the
arc-length methods is dimensionless, see
:func:`.arc_length._solver_arc_length`. ``absTOL`` is in the units of the
forces of the model, ``relTOL`` is relative and, with the default
Crisfield criterion, independent of the units of each degree of freedom,
see :doc:`convergence`.


Log messages and ``silent``
---------------------------

The solvers print their progress to the standard output with the functions
of :mod:`.logger`, indented by level, the warnings starting with
``WARNING:``. Every solver accepts ``silent``, which suppresses the
printing, e.g. ``an.static(NLgeom=True, silent=True)`` or ``lb(K, KG,
silent=True)``. The callables of :class:`.Analysis` must accept the keyword
argument ``silent`` too; the non-linear solvers call them with
``silent=True`` inside the iterations. The
warnings of the eigenvalue solvers that require an action, such as the
:class:`.DenseFallbackWarning`, are also issued with the :mod:`warnings`
module, which ``silent`` does not suppress.
