"""Generalized displacement control

Selected with ``Analysis.NL_method = 'generalized_displacement_control'``.
The implementation is shared with the arc-length methods, see
:mod:`.arc_length`.

"""
from .logger import msg
from .arc_length import _solver_arc_length


def _solver_generalized_displacement_control(an, silent=False):
    r"""Generalized displacement control solver

    Runs :func:`.arc_length._solver_arc_length` with
    ``method='generalized_displacement_control'``.

    Notes
    -----
    The load factor increment of each step is scaled by the square root of
    the generalized stiffness parameter, whose sign reverses the load at
    the limit points, and the corrections are orthogonal to the tangent of
    the previous step, see :func:`.arc_length._solver_arc_length`.

    """

    msg('___________________________________________', level=1, silent=silent)
    msg('                                           ', level=1, silent=silent)
    msg('Generalized displacement control solver', level=1, silent=silent)
    msg('___________________________________________', level=1, silent=silent)
    msg('Initializing...', level=1, silent=silent)
    _solver_arc_length(an, method='generalized_displacement_control',
                       silent=silent)
