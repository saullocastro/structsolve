Linear buckling analysis
========================

The function :func:`.lb` solves the linear buckling eigenvalue problem
`([K] + \lambda [K_G])\{u\} = \{0\}`, returning the load multipliers
`\lambda`.

The example below calculates the critical load of a simply supported
isotropic plate under uniaxial compression, using a Ritz model based on the
first-order shear deformation theory (FSDT). The approximation functions are
the hierarchical Legendre functions of Bardell, the same as those of
`buckling <https://github.com/saullocastro/buckling>`_, and the plate
stiffnesses are those of
`composites <https://github.com/saullocastro/composites>`_; both are
implemented with NumPy in the module ``tests/ritz.py``:

.. literalinclude:: ../../tests/ritz.py

The stiffness matrices are integrated numerically and the critical load is
compared with the analytical solution. The code is extracted from one of the
``structsolve`` unit tests:

.. literalinclude:: ../../tests/test_linear_buckling.py
    :pyobject: test_lb_plate_buckling_fsdt
