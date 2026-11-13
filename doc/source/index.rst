Structural analysis solvers - structsolve
=========================================

The ``structsolve`` module provides structural analysis solvers tailored for
semi-analytical models, such as the Ritz models of plates and shells of
`panels <https://github.com/saullocastro/panels>`_. With ``structsolve`` you
can run:

* Linear static analyses, `\boldsymbol{K} \boldsymbol{u} = \boldsymbol{f}`, with :func:`.static`,
  :func:`.solve` or :class:`.Analysis`

* Linear buckling analyses, `(\boldsymbol{K} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}`, with
  :func:`.lb`

* Frequency analyses, `(\boldsymbol{K} + \lambda^2 \boldsymbol{M}) \boldsymbol{u} = \boldsymbol{0}`, with :func:`.freq`

* Sparse eigenvalue solvers first, with verified results and dense solvers
  only as a last resort, also in the browser (Pyodide) and with the faulty
  Intel MKL versions that break ARPACK, see :doc:`eigensolvers` and
  :doc:`pyodide`

* Non-linear static analyses with :class:`.Analysis`, using:
    - the Newton-Raphson method, full or modified, with an optional
      line-search
    - the arc-length methods of Riks and Crisfield, able to trace limit points,
      snap-through and snap-back

* Configuration-dependent loads, e.g. follower pressures, in the linear and
  non-linear static analyses, see :mod:`.callbacks`, and unsymmetric matrices
  in all the solvers, including the kinetic criterion (flutter) with
  :func:`.freq`

* Load vectors that depend on the load factor, e.g. prescribed
  displacements imposed exactly as linear constraints, in the non-linear
  static analyses, see :mod:`.callbacks`

The constrained degrees of freedom of the models, i.e. the null rows and
columns of the matrices, are removed before solving, see
:doc:`boundary_conditions`.


Code repository
---------------

https://github.com/saullocastro/structsolve


User guide
----------

.. toctree::
    :maxdepth: 1

    conventions.rst
    boundary_conditions.rst
    lb_prestress.rst
    eigensolvers.rst
    eigen_accuracy.rst
    convergence.rst
    sparse_assembly.rst
    pyodide.rst


.. toctree::
    :maxdepth: 1
    :caption: Worked examples

    notebooks/ex_linear_static.nblink
    notebooks/ex_linear_buckling.nblink
    notebooks/ex_frequency.nblink
    notebooks/ex_newton_raphson.nblink
    notebooks/ex_arc_length.nblink
    notebooks/ex_cylinder_postbuckling.nblink
    notebooks/ex_follower_loads.nblink
    notebooks/ex_prescribed_displacements.nblink


.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    notebooks/tutorial_von_mises_truss.nblink
    notebooks/simo1990_cylinder_postbuckling.nblink


structsolve API
---------------

.. toctree::
    :maxdepth: 2

    api.rst


.. toctree::
    :maxdepth: 1
    :caption: About this documentation

    documentation_levels.rst


Installing structsolve
----------------------

Install from the distributed packages by simply doing::

    python -m pip install structsolve

or from the source code using::

    python -m pip install .

In the browser, with `Pyodide <https://pyodide.org>`_, which already
distributes NumPy and SciPy, the wheel published on PyPI is installed with
``micropip``::

    import micropip
    await micropip.install("structsolve")

See :doc:`pyodide`.


.. toctree::
    :maxdepth: 1
    :caption: Changelog

    changelog.rst


License
-------

.. literalinclude:: ../../LICENSE
    :encoding: latin-1


Citing this library
-------------------

Saullo G. P. Castro (2026). Structural analysis solvers tailored for semi-analytical models (Version 0.11.0). Zenodo. DOI: https://doi.org/10.5281/zenodo.2581212.

In BibTeX:

.. code-block:: bibtex

    @misc{castro_structsolve,
      author    = {Castro, Saullo G. P.},
      title     = {Structural analysis solvers tailored for semi-analytical
                   models},
      year      = {2026},
      version   = {0.11.0},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.2581212},
      url       = {https://doi.org/10.5281/zenodo.2581212}
    }

The DOI represents all the versions and resolves to the latest one. The
repository also has a ``CITATION.cff`` file with the same data.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
