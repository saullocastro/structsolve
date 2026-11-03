r"""``Analysis.constraint_values`` with a load vector that depends on the
load factor

The three hardening springs of :mod:`test_load_factor_dependent_load`,
whose node 3 is prescribed exactly, `x_3 = \lambda d_1`, by the null-space
method of `panels`, i.e. with ``calc_fext(inc, c, lbd)`` returning
`-\partial \boldsymbol{R}_r/\partial \lambda = \boldsymbol{Z}^\top
(\boldsymbol{F}_{ext} - \boldsymbol{K}_T \boldsymbol{c}_{p1})`, with a load
`\lambda F` on node 2, and a second prescribed quantity imposed by
``Analysis.constraints`` and ``Analysis.constraint_values`` on the reduced
problem, `\boldsymbol{C}^\top \boldsymbol{c}_r = \lambda e`:

- ``'node1'``: `x_1 = \lambda e`, `\boldsymbol{C} = \{1, 0\}`; node 2 is
  then the only free degree of freedom, in equilibrium when
  `f_2(x_2 - x_1) - f_3(x_3 - x_2) = \lambda F`;
- ``'relative'``: `x_2 - x_1 = \lambda e`, `\boldsymbol{C} = \{-1, 1\}`;
  `x_1` is free, in equilibrium when the sum of the forces on the nodes 1
  and 2 balances the load, `f_1(x_1) - f_3(x_3 - x_2) = \lambda F`.

The single unknown of each case is found by :func:`scipy.optimize.brentq`,
the same problem with both prescribed quantities eliminated by hand.

"""
import numpy as np
import pytest
from scipy.optimize import brentq

from test_load_factor_dependent_load import SpringsProblem

METHODS = ['NR', 'arc_length_riks', 'arc_length_crisfield',
           'displacement_control', 'generalized_displacement_control']

D1 = 1.5
E = -0.5
F = 80.
CASES = {'node1': np.array([1., 0.]), 'relative': np.array([-1., 1.])}


def has_pypardiso():
    try:
        import pypardiso  # noqa: F401
    except ImportError:
        return False
    return True


def spring_force(p, i, e):
    return p.k[i]*e + p.a[i]*e**3


def reference(p, case, lbd):
    """The configuration of the reduced problem at ``lbd``, see the module
    docstring"""
    x3 = lbd*D1
    if case == 'node1':
        x1 = lbd*E

        def g(x2):
            return (spring_force(p, 1, x2 - x1) - spring_force(p, 2, x3 - x2)
                    - lbd*F)
        x2 = brentq(g, -5., 5., xtol=1e-15, rtol=1e-15)
        return np.array([x1, x2])

    def g(x1):
        x2 = x1 + lbd*E
        return spring_force(p, 0, x1) - spring_force(p, 2, x3 - x2) - lbd*F
    x1 = brentq(g, -5., 5., xtol=1e-15, rtol=1e-15)
    return np.array([x1, x1 + lbd*E])


def make(case, method, **kwargs):
    p = SpringsProblem(d0=0., d1=D1)
    p.Fext = np.array([0., F, 0.])
    if method == 'displacement_control':
        # the free degree of freedom
        kwargs['control_vector'] = (np.array([0., 1.]) if case == 'node1'
                                    else np.array([1., 0.]))
    an = p.analysis(NL_method=method, constraints=CASES[case],
                    constraint_values=[E], initialInc=0.1, **kwargs)
    return p, an


@pytest.mark.parametrize('case', list(CASES))
@pytest.mark.parametrize('method', METHODS)
def test_path(method, case):
    p, an = make(case, method)
    an.static(NLgeom=True, silent=True)
    assert an.increments[-1] == 1.
    assert len(an.increments) > 3
    # calc_fext received the load factor
    assert p.fext_calls and any(lbd for lbd, _ in p.fext_calls)
    C = CASES[case]
    for lbd, c in zip(an.increments, an.cs):
        assert abs(C @ c - lbd*E) <= 1e-12
        np.testing.assert_allclose(c, reference(p, case, lbd), rtol=1e-8,
                                   atol=1e-12)
        # the residual of the free direction vanishes
        Z = np.array([1., 1.]) if case == 'relative' else np.array([0., 1.])
        R = p.residual(c, lbd)
        assert abs(Z @ R) <= 1e-7*np.abs(p.load_vector(c, lbd)).max()
    # far from the linear springs
    lin = reference(SpringsProblem(a=(0., 0., 0.)), case, 1.)
    assert np.abs(an.cs[-1] - lin).max() > 0.05


@pytest.mark.skipif(not has_pypardiso(), reason='pypardiso is not installed')
@pytest.mark.parametrize('method', ['NR', 'arc_length_riks'])
def test_pardiso(method):
    out = []
    for linear_solver in ('splu', 'pardiso'):
        p, an = make('relative', method, linear_solver=linear_solver)
        an.static(NLgeom=True, silent=True)
        out.append((np.asarray(an.increments), np.asarray(an.cs)))
    np.testing.assert_allclose(out[1][0], out[0][0], rtol=1e-10)
    np.testing.assert_allclose(out[1][1], out[0][1], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize('method', ['NR', 'arc_length_riks'])
def test_modified_newton_raphson(method):
    p, an = make('node1', method, modified_NR=True)
    an.static(NLgeom=True, silent=True)
    assert an.increments[-1] == 1.
    np.testing.assert_allclose(an.cs[-1], reference(p, 'node1', 1.),
                               rtol=1e-8)
