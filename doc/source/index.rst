Structural analysis solvers - structsolve
=========================================

The ``structsolve`` module provides structural analysis solvers tailored for
semi-analytical models, such as the Ritz models of plates and shells of
`panels <https://github.com/saullocastro/panels>`_. With ``structsolve`` you
can run:

* Linear static analyses, `[K]\{u\} = \{f\}`, with :func:`.static`,
  :func:`.solve` or :class:`.Analysis`

* Linear buckling analyses, `([K] + \lambda [K_G])\{u\} = \{0\}`, with
  :func:`.lb`

* Frequency analyses, `([K] + \lambda^2 [M])\{u\} = \{0\}`, with :func:`.freq`

* Non-linear static analyses with :class:`.Analysis`, using:
    - the Newton-Raphson method, full or modified, with an optional
      line-search
    - the arc-length methods of Riks and Crisfield, able to trace limit points,
      snap-through and snap-back

The constrained degrees of freedom of the models, i.e. the null rows and
columns of the matrices, are removed before solving.


Code repository
---------------

https://github.com/saullocastro/structsolve


Citing this library
-------------------

Saullo G. P. Castro (2026). Structural analysis solvers tailored for semi-analytical models (Version 0.4.3). Zenodo. DOI: https://doi.org/10.5281/zenodo.2581212.


Usage examples
--------------

.. toctree::
    :maxdepth: 1

    ex_linear_static.rst
    ex_linear_buckling.rst
    ex_frequency.rst
    ex_newton_raphson.rst
    ex_arc_length.rst


structsolve API
---------------

.. toctree::
    :maxdepth: 2

    api.rst


Installing structsolve
----------------------

Install from the distributed packages by simply doing::

    python -m pip install structsolve

or from the source code using::

    python -m pip install .


Changelog
---------

https://github.com/saullocastro/structsolve/blob/main/CHANGELOG.md


License
-------

.. literalinclude:: ../../LICENSE
    :encoding: latin-1


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
