Github Actions status:
[![pytest](https://github.com/compmech/structsolve/actions/workflows/pytest.yml/badge.svg)](https://github.com/compmech/structsolve/actions/workflows/pytest.yml)
[![Deploy](https://github.com/compmech/structsolve/actions/workflows/pythonpublish.yml/badge.svg)](https://github.com/compmech/structsolve/actions/workflows/pythonpublish.yml)

Coverage status:
[![codecov](https://github.com/compmech/structsolve/actions/workflows/coverage.yml/badge.svg)](https://github.com/compmech/structsolve/actions/workflows/coverage.yml)
[![Codecov Status](https://codecov.io/gh/compmech/structsolve/branch/master/graph/badge.svg)](https://codecov.io/gh/compmech/structsolve)


Structural analysis solvers tailored for semi-analytical models
===============================================================

- Linear statics: [K]{u} = {f}
- Eigensolver for Linear buckling: ([K] + lambda[KG]){u} = 0
- Eigensolver for dynamics: ([K] + lambda^2[M]){u} = 0
- Nonlinear statics using Newton-Raphson 
- Nonlinear statics using the Arc-Length method

Currently these solvers are pretty much compatible with my other repositories
[panels](https://github.com/saullocastro/panels), 
[buckling](https://github.com/saullocastro/buckling).


License
-------
Distrubuted in the 3-Clause BSD license (https://raw.github.com/compmech/structsolve/master/LICENSE).

Contact: S.G.P.Castro@tudelft.nl

