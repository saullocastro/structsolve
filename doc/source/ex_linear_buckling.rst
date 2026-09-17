Linear buckling analysis
========================

The function :func:`.lb` solves the linear buckling eigenvalue problem
`([K] + \lambda [K_G])\{u\} = \{0\}`, returning the load multipliers
`\lambda`.

The example below calculates the critical load of a simply supported
isotropic plate under uniaxial compression, using a Ritz model based on the
first-order shear deformation theory (FSDT), with the Legendre polynomials of
`buckling <https://github.com/saullocastro/buckling>`_ as approximation
functions and the plate properties of
`composites <https://github.com/saullocastro/composites>`_. The stiffness
matrices are integrated numerically and the critical load is compared with
the analytical solution. The code is extracted from one of the
``structsolve`` unit tests:

.. literalinclude:: ../../tests/test_linear_buckling.py
    :pyobject: test_lb_plate_buckling_fsdt
