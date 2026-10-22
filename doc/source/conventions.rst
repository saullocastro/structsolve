Notation and sign conventions
=============================

Static analyses
---------------

`\{c\}` is the vector of the degrees of freedom of the model, the solution
vectors of ``an.cs``, and `\lambda` the load factor, the values of
``an.increments``. The residual force vector of the non-linear analyses is

.. math::

    \{R(c, \lambda)\} = \lambda \{F_{ext}\} - \{F_{int}(c)\}

with `\{F_{ext}\}` = ``calc_fext(inc=1.)`` and `\{F_{int}\}` =
``calc_fint(c)``, and the tangent stiffness matrix is

.. math::

    [K_T] = [K_C] + [K_G] = \frac{\partial F_{int}}{\partial c}
          = -\frac{\partial R}{\partial c}

with `[K_C]` = ``calc_kC(c, NLgeom=True)`` and `[K_G]` = ``calc_kG(c,
NLgeom=True)``. A configuration-dependent load `\{F(c)\}` has the load
stiffness matrix `-\partial F/\partial c`, which enters `[K_T]`, see
:mod:`.callbacks`. The linear static analysis solves `[K_0]\{c\} =
\{F_{ext}\}`, with `[K_0]` = ``calc_kC()``.


Eigenvalue analyses
-------------------

.. list-table::
    :header-rows: 1

    * - Analysis
      - Eigenvalue problem
      - Returned eigenvalues
    * - :func:`.lb`
      - `([K] + \lambda [K_G])\{u\} = \{0\}`
      - load multipliers `\lambda`, the positive ones first, in increasing
        order
    * - :func:`.freq`
      - `([K] + \lambda^2 [M])\{u\} = \{0\}`
      - ``lambda2`` `= -\omega_n^2`, negative for a stable structure

For :func:`.lb`, `[K_G]` is the geometric stiffness matrix of the
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
forces of the model, ``relTOL`` is relative, see :doc:`convergence`.


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
