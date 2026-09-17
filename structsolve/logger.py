"""Logging functions used by the solvers"""


def warn(msg, level=0, silent=False):
    """Print a warning message, starting with ``WARNING:``

    Parameters
    ----------
    msg : str
        The message.
    level : int, optional
        Indentation level, each level adds a tab.
    silent : bool, optional
        If ``True`` the message is not printed.

    Returns
    -------
    msg : str
        The message.

    """
    msg = 'WARNING: ' + msg
    if not silent:
        print('\t'*level + msg)
    return msg


def error(msg, level=0, silent=False):
    """Print an error message, starting with ``ERROR:``

    Parameters
    ----------
    msg : str
        The message.
    level : int, optional
        Indentation level, each level adds a tab.
    silent : bool, optional
        If ``True`` the message is not printed.

    Returns
    -------
    msg : str
        The message.

    """
    msg = 'ERROR: ' + msg
    if not silent:
        print('\t'*level + msg)
    return msg


def msg(msg, level=0, silent=False):
    """Print a log message

    Parameters
    ----------
    msg : str
        The message.
    level : int, optional
        Indentation level, each level adds a tab.
    silent : bool, optional
        If ``True`` the message is not printed.

    Returns
    -------
    msg : str
        The message.

    """
    if not silent:
        print('\t'*level + msg)
    return msg
