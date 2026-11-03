r"""Tests of ``Analysis.inertia_num_eigvals``: the number of negative real
eigenvalues of an unsymmetric tangent stiffness matrix

The count is that of the eigenvalues of smallest magnitude computed by
shift-invert with ARPACK and the SuperLU factorization of the converged
state, see :func:`.newton_raphson._eigvals_inertia`, checked against
:func:`numpy.linalg.eigvals` of the dense matrix.

The problem is linear in the configuration, with a tangent stiffness matrix
that depends on the load factor, as with a follower load whose load
stiffness is scaled by the load factor:

.. math::

    \boldsymbol{F}_{int}(\boldsymbol{c}, \lambda) = \boldsymbol{K}(\lambda)
    \boldsymbol{c}, \qquad \boldsymbol{K}(\lambda) =
    \mathrm{diag}\left(k_0, \boldsymbol{S} (\boldsymbol{D}_0 + \lambda
    \boldsymbol{D}_1) \boldsymbol{S}^{-1}\right), \qquad
    \boldsymbol{F}_{ext} = \boldsymbol{e}_0

with the load on the degree of freedom 0, uncoupled from the others, such
that the path is `c_0 = \lambda/k_0`, the other degrees of freedom null,
and the eigenvalues of `\boldsymbol{K}(\lambda)` are those of the block
diagonal `\boldsymbol{D}_0 + \lambda \boldsymbol{D}_1`: real ones linear in
`\lambda`, of which some cross zero at given load factors, and complex pairs
`a(\lambda) \pm i b` of `2 \times 2` blocks. `\boldsymbol{S}` is a random
non-orthogonal matrix, which makes `\boldsymbol{K}` unsymmetric and
non-normal.

"""
import numpy as np
import pytest
from scipy.linalg import block_diag, null_space
from scipy.sparse import csr_matrix

from structsolve import Analysis
from structsolve.static import _factorize, _count_negative_real
from structsolve.pardiso import available as pardiso_available


def has_pardiso():
    return pardiso_available()


class LinearUnsymmetric(object):
    """See the module docstring: ``crossings`` are the load factors at which
    the real eigenvalues cross zero, ``complex_crossings`` those at which
    the real part of the complex pairs does, ``num_positive`` real
    eigenvalues stay positive"""
    def __init__(self, crossings, complex_crossings=(), num_positive=20,
                 seed=0, k0=10.):
        rng = np.random.default_rng(seed)
        blocks0, blocks1 = [], []
        for i, lc in enumerate(crossings):
            d0 = 1. + 0.5*i
            blocks0.append([[d0]])
            blocks1.append([[-d0/lc]])
        for lc in complex_crossings:
            a0, b = 1.5, 2.
            blocks0.append([[a0, b], [-b, a0]])
            blocks1.append(-a0/lc*np.eye(2))
        for i in range(num_positive):
            d = 3. + i
            blocks0.append([[d]])
            blocks1.append([[0.5*rng.uniform(-1., 1.)]])
        D0 = block_diag(*blocks0)
        D1 = block_diag(*blocks1)
        nb = D0.shape[0]
        S = np.eye(nb) + 0.3*rng.standard_normal((nb, nb))
        Sinv = np.linalg.inv(S)
        self.B0 = S @ D0 @ Sinv
        self.B1 = S @ D1 @ Sinv
        self.k0 = k0
        self.n = nb + 1
        self.an = None

    def K(self, lbd):
        return block_diag([[self.k0]], self.B0 + lbd*self.B1)

    def calc_fext(self, inc=1., silent=True):
        f = np.zeros(self.n)
        f[0] = inc
        return f

    def calc_fint(self, c, inc=1., silent=True):
        return self.K(inc) @ c

    def calc_kC(self, c=None, NLgeom=False, inc=0., silent=True):
        return csr_matrix(self.K(inc))

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        return csr_matrix((self.n, self.n))

    def analysis(self, **kwargs):
        an = Analysis(self.calc_fext, self.calc_fint, self.calc_kC,
                      self.calc_kG)
        for key, value in kwargs.items():
            setattr(an, key, value)
        self.an = an
        return an


def dense_count(A, k=None):
    """Number of negative real eigenvalues of the dense ``A``, among the
    ``k`` of smallest magnitude when given"""
    lam = np.linalg.eigvals(A)
    lam = lam[np.argsort(np.abs(lam))]
    if k is not None:
        lam = lam[:k]
    return _count_negative_real(lam)[0]


@pytest.mark.parametrize('k', [1, 3, 6, 40])
@pytest.mark.parametrize('seed', range(3))
def test_smallest_eigvals(seed, k):
    """The eigenvalues of ARPACK and of the dense path against
    numpy.linalg.eigvals, with null rows and columns"""
    p = LinearUnsymmetric([0.2, 0.5], complex_crossings=[0.4], seed=seed)
    for lbd in (0.1, 0.45, 0.7):
        A = np.zeros((p.n + 2, p.n + 2))
        A[2:, 2:] = p.K(lbd)
        f = _factorize(csr_matrix(A))
        lam = f.smallest_eigvals(k)
        ref = np.linalg.eigvals(p.K(lbd))
        ref = ref[np.argsort(np.abs(ref))][:k]
        assert lam.shape == (min(k, p.n),)
        np.testing.assert_allclose(np.sort(np.abs(lam)), np.sort(np.abs(ref)),
                                   rtol=1e-8)
        assert _count_negative_real(lam) == _count_negative_real(ref)


@pytest.mark.parametrize('k', [3, 40])
def test_smallest_eigvals_constraints(k):
    """With constraints, the eigenvalues of ``Z^T K Z``, ``Z`` an
    orthonormal basis of the null space of ``C^T``"""
    p = LinearUnsymmetric([0.2, 0.5], complex_crossings=[0.4], seed=1)
    rng = np.random.default_rng(5)
    C = rng.standard_normal((p.n, 2))
    Z = null_space(C.T)
    for lbd in (0.1, 0.45, 0.7):
        f = _factorize(csr_matrix(p.K(lbd)), constraints=C)
        lam = f.smallest_eigvals(k)
        ref = np.linalg.eigvals(Z.T @ p.K(lbd) @ Z)
        ref = ref[np.argsort(np.abs(ref))][:k]
        np.testing.assert_allclose(np.sort(np.abs(lam)), np.sort(np.abs(ref)),
                                   rtol=1e-8)
        assert _count_negative_real(lam) == _count_negative_real(ref)


METHODS = ['NR', 'arc_length_riks', 'arc_length_crisfield']


@pytest.mark.parametrize('linear_solver',
                         ['splu'] + (['pardiso'] if has_pardiso() else []))
@pytest.mark.parametrize('method', METHODS)
def test_count_along_path(method, linear_solver):
    """Two real eigenvalues cross zero, 0 -> 1 -> 2, the sign of the
    determinant returning to +1, and a complex pair crosses the imaginary
    axis, which is not counted"""
    p = LinearUnsymmetric([0.27, 0.63], complex_crossings=[0.47])
    an = p.analysis(NL_method=method, inertia=True, inertia_num_eigvals=6,
                    linear_solver=linear_solver, initialInc=0.1, maxInc=0.1)
    with pytest.warns(RuntimeWarning) if linear_solver == 'pardiso' \
            else _nowarn():
        an.static(NLgeom=True, silent=True)
    assert an.increments[-1] == 1.
    counts = an.num_negative_eigvals
    assert len(counts) == len(an.increments)
    for lbd, c, nneg, det in zip(an.increments, an.cs, counts, an.det_signs):
        np.testing.assert_allclose(c[0], lbd/p.k0, rtol=1e-12)
        assert nneg == dense_count(p.K(lbd), 6) == dense_count(p.K(lbd))
        assert nneg == int(lbd > 0.27) + int(lbd > 0.63)
        assert det == (-1)**nneg
    assert sorted(set(counts)) == [0, 1, 2]
    assert an.det_signs[-1] == 1
    # the complex pair has a negative real part at the end
    lam = np.linalg.eigvals(p.K(1.))
    assert np.any((lam.real < 0) & (np.abs(lam.imag) > 1.))
    assert an.num_perturbed_pivots == [None]*len(counts)


class _nowarn(object):
    def __enter__(self):
        import warnings
        self.cm = warnings.catch_warnings()
        self.cm.__enter__()
        warnings.simplefilter('error')

    def __exit__(self, *args):
        return self.cm.__exit__(*args)


@pytest.mark.parametrize('method', METHODS)
def test_max_inertia_jump(method):
    """Two eigenvalues crossing zero within an increment: the increment is
    repeated until they cross one at a time"""
    p = LinearUnsymmetric([0.25, 0.27], seed=2)
    an = p.analysis(NL_method=method, inertia=True, inertia_num_eigvals=4,
                    max_inertia_jump=1, initialInc=0.1)
    an.static(NLgeom=True, silent=True)
    assert an.increments[-1] == 1.
    counts = an.num_negative_eigvals
    assert counts[-1] == 2
    assert np.all(np.abs(np.diff([0] + counts)) <= 1)
    assert 1 in counts
    # without the check the two cross in the same increment
    an = p.analysis(NL_method=method, inertia=True, inertia_num_eigvals=4,
                    initialInc=0.1)
    an.static(NLgeom=True, silent=True)
    assert 1 not in an.num_negative_eigvals


def test_incomplete_count_warns(capsys):
    """Three negative real eigenvalues, the most negative beyond the k of
    smallest magnitude: the parity differs from the sign of the
    determinant, which is reported"""
    p = LinearUnsymmetric([0.2, 0.3, 0.9], num_positive=10, seed=3)
    # at lbd = 1 the real eigenvalues are 1 - 1/0.2 = -4, 1.5 - 1.5/0.3 =
    # -3.5 and 2 - 2/0.9 = -0.22, the positive ones 3 + i + O(0.5)
    lam = np.linalg.eigvals(p.K(1.))
    assert dense_count(p.K(1.)) == 3
    k = int(np.count_nonzero(np.abs(lam) < 3.75))
    assert dense_count(p.K(1.), k) == 2
    an = p.analysis(NL_method='NR', inertia=True, inertia_num_eigvals=k,
                    initialInc=0.5)
    an.static(NLgeom=True, silent=False)
    assert 'odd number of negative real eigenvalues' in capsys.readouterr().out
    assert an.num_negative_eigvals[-1] == 2
    assert an.det_signs[-1] == -1


def test_symmetric_splu_matches_pardiso():
    """On a symmetric tangent with ``'splu'``, the count of all the
    eigenvalues (``k`` larger than the size) is the inertia"""
    from analytic_problems import von_mises_truss
    p = von_mises_truss(frac=1.5, ks=0.5)
    p.analysis(NL_method='arc_length_riks', initialInc=0.05, inertia=True,
               inertia_num_eigvals=2)
    p.an.static(NLgeom=True, silent=True)
    counts = p.an.num_negative_eigvals
    assert counts == [(1 - s)//2 for s in p.an.det_signs]
    assert 1 in counts


def test_default_and_validation():
    p = LinearUnsymmetric([0.27, 0.63])
    an = p.analysis(NL_method='NR', inertia=True)
    assert an.inertia_num_eigvals is None
    an.static(NLgeom=True, silent=True)
    assert an.num_negative_eigvals == [None]*len(an.increments)
    for kw, match in [(dict(inertia_num_eigvals=2), 'requires'),
                      (dict(inertia=True, inertia_num_eigvals=0), 'positive'),
                      (dict(inertia=True, inertia_num_eigvals=1.5),
                       'positive'),
                      (dict(inertia=True, max_inertia_jump=1),
                       'inertia_num_eigvals')]:
        an = p.analysis(NL_method='arc_length_riks', **kw)
        with pytest.raises(ValueError, match=match):
            an.static(NLgeom=True, silent=True)
