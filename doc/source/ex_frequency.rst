Frequency analysis
==================

The function :func:`.freq` solves the eigenvalue problem
`([K] + \lambda^2 [M])\{u\} = \{0\}`, where `\lambda^2 = -\omega_n^2`
and `\omega_n` is the natural frequency in rad/s. The code below is extracted
from one of the ``structsolve`` unit tests:

.. literalinclude:: ../../tests/test_freq.py
    :pyobject: test_freq_simple
