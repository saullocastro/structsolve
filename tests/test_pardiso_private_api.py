"""Tests of the fallback of the phase 22 of PARDISO to the public API

:func:`.static._pardiso_factorize` reuses the analysis of PARDISO with the
private API of ``pypardiso`` (checked against 0.4.7); when it is missing,
has another signature, raises or gives another factorization, the public
``factorize`` (phase 12) is used, with one warning, and the results are
the same.

"""
import importlib
import warnings

import numpy as np
import pytest

from analytic_problems import von_mises_truss

pypardiso = pytest.importorskip('pypardiso')

#NOTE structsolve.static is the function static(), exported by the package
static_module = importlib.import_module('structsolve.static')

Solver = pypardiso.PyPardisoSolver


def run(monkeypatch, method):
    """The snap-through of the von Mises truss with PARDISO and the inertia,
    the private API checked anew; the results, the warnings of the
    private API and its final state"""
    monkeypatch.setattr(static_module, '_pardiso_private_api', None)
    phases = []
    factorize = static_module._pardiso_factorize

    def counted(solver, A, phase):
        phases.append(phase)
        return factorize(solver, A, phase)

    monkeypatch.setattr(static_module, '_pardiso_factorize', counted)
    p = von_mises_truss(frac=1.5, ks=0.5)
    p.analysis(NL_method=method, initialInc=0.05, inertia=True,
               linear_solver='pardiso')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        p.an.static(NLgeom=True, silent=True)
    private = [x for x in w if 'private API' in str(x.message)]
    an = p.an
    assert 22 in phases
    return (an.increments, np.asarray(an.cs), an.num_negative_eigvals,
            an.num_perturbed_pivots), private, \
        static_module._pardiso_private_api


def signature_changed(monkeypatch):
    call = Solver._call_pardiso

    def _call_pardiso(self, A, b, nrhs=None):
        return call(self, A, b)

    monkeypatch.setattr(Solver, '_call_pardiso', _call_pardiso)


def missing(monkeypatch):
    monkeypatch.delattr(Solver, '_hash_csr_matrix')


def raises(monkeypatch):
    def set_phase(self, phase):
        if phase == 22:
            raise RuntimeError('no phase 22')
        self.phase = phase

    monkeypatch.setattr(Solver, 'set_phase', set_phase)


def does_nothing(monkeypatch):
    """The phase 22 silently skipped: no factorization"""
    call = Solver._call_pardiso

    def _call_pardiso(self, A, b):
        if self.phase == 22:
            return np.zeros_like(b)
        return call(self, A, b)

    monkeypatch.setattr(Solver, '_call_pardiso', _call_pardiso)


def wrong_phase(monkeypatch):
    """The phase 22 replaced by a factorization of another matrix: no
    exception, a wrong factorization, found by the verification"""
    call = Solver._call_pardiso

    def _call_pardiso(self, A, b):
        if self.phase == 22:
            A = A.copy()
            A.data = 2*A.data
        return call(self, A, b)

    monkeypatch.setattr(Solver, '_call_pardiso', _call_pardiso)


@pytest.mark.parametrize('method', ['arc_length_riks', 'NR'])
@pytest.mark.parametrize('breakage', [signature_changed, missing, raises,
                                      does_nothing, wrong_phase])
def test_fallback(monkeypatch, breakage, method):
    ref, private, state = run(monkeypatch, method)
    assert private == []
    assert state is True
    with monkeypatch.context() as m:
        breakage(m)
        out, private, state = run(m, method)
        assert state is False
        assert len(private) == 1
        assert issubclass(private[0].category, RuntimeWarning)
    assert out[0] == ref[0]
    np.testing.assert_allclose(out[1], ref[1], rtol=1e-12, atol=1e-15)
    assert out[2] == ref[2]
    assert out[3] == ref[3]
    if method == 'arc_length_riks':
        assert 1 in ref[2]


def test_factorization_failure_not_the_api(monkeypatch):
    """An exception of the phase 12 as well is a failure of the
    factorization: raised, the private API kept"""
    from scipy.sparse import csr_matrix
    monkeypatch.setattr(static_module, '_pardiso_private_api', True)
    solver = Solver(mtype=-2)
    A = csr_matrix(np.array([[2., 1.], [0., 3.]]))
    solver.factorize(A)
    B = csr_matrix(np.array([[2., 1.], [0., 0.]]))
    # an empty row: pypardiso raises in _check_A, both ways
    B.eliminate_zeros()
    with pytest.raises(ValueError):
        static_module._pardiso_factorize(solver, B, 22)
    assert static_module._pardiso_private_api is True
    static_module._pardiso_release(solver)
