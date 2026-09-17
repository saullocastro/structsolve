from .logger import msg
from .arc_length import _solver_arc_length


def _solver_arc_length_crisfield(an, silent=False):
    r"""Arc-Length solver using Crisfield`s method

    The quadratic (spherical) arc-length constraint is solved exactly at
    every iteration, see :func:`.arc_length._solver_arc_length`.

    """
    msg('___________________________________________', level=1, silent=silent)
    msg('                                           ', level=1, silent=silent)
    msg('Arc-Length solver using Crisfield implementation', level=1, silent=silent)
    msg('___________________________________________', level=1, silent=silent)
    msg('Initializing...', level=1, silent=silent)
    _solver_arc_length(an, method='crisfield', silent=silent)
