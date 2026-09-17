r"""
===============================================
Structural Analysis Solver (:mod:`structsolve`)
===============================================

.. currentmodule:: structsolve

Structural analysis solvers tailored for semi-analytical models, available
directly from the ``structsolve`` namespace:

- :class:`.Analysis`: linear and non-linear static analyses, using the
  Newton-Raphson or the arc-length methods
- :func:`.solve`: solution of linear systems removing null rows and columns
- :func:`.static`: linear static analysis
- :func:`.lb`: linear buckling analysis
- :func:`.freq`: frequency analysis

"""
from __future__ import absolute_import

from .analysis import Analysis
from .freq import freq
from .linear_buckling import lb
from .static import solve, static
