from .logger import msg
from .arc_length import _solver_arc_length


def _solver_arc_length_riks(an, silent=False):
    r"""Arc-Length solver using the Riks method

    The arc-length constraint is linearized at every iteration (updated
    normal plane), see :func:`.arc_length._solver_arc_length`.

    """
    msg('___________________________________________', level=1, silent=silent)
    msg('                                           ', level=1, silent=silent)
    msg('Arc-Length solver using Riks implementation', level=1, silent=silent)
    msg('___________________________________________', level=1, silent=silent)
    msg('Initializing...', level=1, silent=silent)
    _solver_arc_length(an, method='riks', silent=silent)
