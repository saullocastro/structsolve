"""Indirect displacement control

Selected with ``Analysis.NL_method = 'displacement_control'`` and the
control vector ``Analysis.control_vector``. The implementation is shared
with the arc-length methods, see :mod:`.arc_length`.

"""
from .logger import msg
from .arc_length import _solver_arc_length


def _solver_displacement_control(an, silent=False):
    r"""Indirect displacement control solver

    Runs :func:`.arc_length._solver_arc_length` with
    ``method='displacement_control'``.

    Notes
    -----
    The arc-length constraint is replaced by the linear constraint
    `\boldsymbol{\psi}^\top \Delta \boldsymbol{c} = \Delta s`, with
    `\boldsymbol{\psi}` = ``an.control_vector``, see
    :func:`.arc_length._solver_arc_length`.

    """

    msg('___________________________________________', level=1, silent=silent)
    msg('                                           ', level=1, silent=silent)
    msg('Indirect displacement control solver', level=1, silent=silent)
    msg('___________________________________________', level=1, silent=silent)
    msg('Initializing...', level=1, silent=silent)
    _solver_arc_length(an, method='displacement_control', silent=silent)
