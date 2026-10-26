Linear buckling with a prestress
================================

A linear buckling analysis with :func:`.lb` has three steps, of which only
the last one is done by ``structsolve``:

1. the linear static solution `\boldsymbol{c}_0` of a reference load
   `\boldsymbol{F}_{ref}`, `\boldsymbol{K}_{C0} \boldsymbol{c}_0 = \boldsymbol{F}_{ref}`, e.g. with :func:`.solve`,
   :func:`.static` or ``Analysis.static(NLgeom=False)``;
2. the geometric stiffness matrix `\boldsymbol{K}_G(\boldsymbol{c}_0)` of the prestress of
   `\boldsymbol{c}_0`, computed by the model, e.g. from the pre-buckling stress
   resultants at each integration point;
3. the eigenvalue problem

   .. math::

       (\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G(\boldsymbol{c}_0)) \boldsymbol{u} = \boldsymbol{0}

   solved by ``lb(K0, KG)``, which returns the load multipliers `\lambda`
   and the buckling modes `\boldsymbol{u}`.

Since the static solution is linear, `\boldsymbol{K}_G` is proportional to the load,
`\boldsymbol{K}_G(\lambda \boldsymbol{c}_0) = \lambda \boldsymbol{K}_G(\boldsymbol{c}_0)`, and the eigenvalue problem gives
the load levels `\lambda \boldsymbol{F}_{ref}` at which `\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G` is
singular.


Meaning of the load multipliers
-------------------------------

`\lambda` multiplies the reference load: the critical load is `\lambda_1
\boldsymbol{F}_{ref}`, and a reference load `\alpha` times larger gives load
multipliers `\alpha` times smaller. A load that is kept constant, e.g. a
preload `\boldsymbol{F}_0`, is not multiplied by `\lambda`: its geometric stiffness
matrix belongs to the first matrix, ``lb(K0 + KG0, KG)``, and the critical
load is `\boldsymbol{F}_0 + \lambda_1 \boldsymbol{F}_{ref}`.


Negative and complex load multipliers
-------------------------------------

A negative `\lambda` is a buckling load of the reversed reference load,
e.g. a plate under tension buckles only under the opposite, compressive,
load. :func:`.lb` returns the positive load multipliers first, in
increasing order, followed, when fewer than ``num_eigvalues`` positive ones
exist, by the negative load multipliers closest to zero, i.e. the critical
load multipliers of the reversed load, see :doc:`eigensolvers`. A reference
load without any positive load multiplier, e.g. a tensile load, thus gives
the critical load multipliers of the reversed load directly. The null rows
of ``KG``, e.g. of the in-plane dofs, give infinite load multipliers, which
the sparse solvers do not return.

Complex load multipliers only exist for unsymmetric matrices, e.g. with the
load stiffness of follower loads, for which :func:`.lb` returns the lowest
positive real load multiplier, see :doc:`ex_follower_loads`.


Example
-------

The plate of `panels <https://github.com/saullocastro/panels>`_ under a
uniaxial compression, whose matrices ``K`` and ``KG`` are saved in
``tests/data``: a reference load twice as large, a constant preload, and the
reversed, tensile, reference load, which has only negative load
multipliers:

.. literalinclude:: ../../tests/test_linear_buckling.py
    :pyobject: test_lb_reference_load

The negative load multipliers closest to zero after the positive ones, for
the sparse and the dense solvers:

.. literalinclude:: ../../tests/test_linear_buckling.py
    :pyobject: test_lb_negative_multipliers_closest_to_zero

A complete Ritz model of a plate, in which the geometric stiffness matrix
is integrated from the stress resultants of the reference load, is in
:doc:`ex_linear_buckling`.
