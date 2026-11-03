r"""Post-buckling of an axially compressed cylinder, Simo, Fox and Rifai (1990)

Utilities of the notebook ``simo1990_cylinder_postbuckling.ipynb`` and of
the test ``tests/test_simo1990_cylinder.py``, and the command line used for
the long convergence runs of the documentation page
``doc/source/ex_cylinder_postbuckling.rst``.

Benchmark of the Riks method of structsolve deep into the post-buckled
regime: Section 6.3.4 of Simo, Fox and Rifai (1990), "Post-buckling analysis
of a simply-supported cylinder; effect of initial imperfections", pp. 54-57,
Fig. 6.3.4 on p. 56. A cylinder of radius ``R = 36``, length ``L = 100``,
thickness ``t = 0.125``, ``E = 1e7``, ``nu = 0.3`` under axial compression;
an ``L/10 x 15 deg`` section is modelled, with symmetry conditions on the
interior edges and a simple support on the loaded edge, such that the
diamond mode of 5 axial half-waves and 12 circumferential waves is the
lowest one. The section is a single :class:`panels.shell.Shell` of the
model ``'cylshell_clpt_donnell'`` (panels 0.15.0) with Bardell's functions.

Neither panels nor structsolve takes a geometric imperfection, which is
added here exactly for the Donnell kinematics with the Ritz vector ``c0`` of
the imperfection, see :class:`ImperfectShell` and the documentation page.

panels (which requires structsolve>=0.10.0) is not a dependency of
structsolve. Run the command line with structsolve and panels 0.15.0
importable, and oneMKL for 'pardiso', from the root of the repository::

    PYTHONPATH=".;<panels>" python notebooks/simo1990_utils.py verify
    PYTHONPATH=... python notebooks/simo1990_utils.py lb 12 12
    PYTHONPATH=... python notebooks/simo1990_utils.py riks 0.1 12 12
    PYTHONPATH=... python notebooks/simo1990_utils.py riks 0.5 12 12
    PYTHONPATH=... python notebooks/simo1990_utils.py perfect 8 8
    PYTHONPATH=... python notebooks/simo1990_utils.py perfect 12 12 10 pardiso
    PYTHONPATH=... python notebooks/simo1990_utils.py summary
    PYTHONPATH=... python notebooks/simo1990_utils.py plot

(``;`` separates the paths on Windows, ``:`` elsewhere.) The commands
``riks`` and ``perfect`` save ``simo1990_<case>_m<m>n<n>.npz`` in the
directory given by the environment variable ``SIMO1990_OUT`` (default: the
current directory), read by ``summary`` and ``plot``, which writes
``ex_cylinder_postbuckling.png``.

"""
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from structsolve import Analysis, lb, solve
from structsolve.stopping import value_reached

from panels.shell import Shell
from panels.bardell import calc_vec_f, calc_vec_fp

R = 36.
L = 100.
t = 0.125
E = 1.e7
nu = 0.3
A_SEC = L/10. # axial length of the section
B_SEC = R*np.pi/12. # arc length of the 15 deg section
#: classical buckling load per unit length of the circumference
N_CL = E*t**2/(R*np.sqrt(3*(1 - nu**2)))

OUT = os.environ.get('SIMO1990_OUT', '.')

# Values read from Fig. 6.3.4 of Simo, Fox and Rifai (1990), p. 56, on the
# page rendered at 300 dpi with a grid of 0.001 x 0.05 drawn over the axes;
# reading precision about +-0.0003 in the axial deflection and +-0.01 in the
# load normalized by the linear buckling load. (axial deflection, load)
#: Fig. (a), triangles, 9-ANS of Stanley (1985), 10 % imperfection
FIG_A_ANS_10 = [
    (0.0071, 0.331), (0.0095, 0.445), (0.0125, 0.570), (0.0148, 0.666),
    (0.0159, 0.717), (0.0166, 0.713), (0.0163, 0.693), (0.0157, 0.653),
    (0.0152, 0.619), (0.0148, 0.580), (0.0144, 0.532), (0.0141, 0.487),
    (0.0140, 0.430), (0.0143, 0.382), (0.0157, 0.346), (0.0184, 0.319),
    (0.0218, 0.324), (0.0296, 0.349), (0.0353, 0.395)]
#: Fig. (a), diamonds, Stanley (1985), 50 % imperfection
FIG_A_ANS_50 = [
    (0.0055, 0.229), (0.0066, 0.273), (0.0077, 0.310), (0.0084, 0.342),
    (0.0089, 0.366), (0.0094, 0.383), (0.0098, 0.397), (0.0106, 0.402),
    (0.0109, 0.385), (0.0112, 0.367), (0.0118, 0.354), (0.0122, 0.332),
    (0.0132, 0.312), (0.0151, 0.300), (0.0175, 0.297), (0.0204, 0.300),
    (0.0237, 0.318), (0.0265, 0.349), (0.0363, 0.420)]
#: Fig. (a), solid line, Simo et al., 10 % imperfection
FIG_A_SIMO_10 = [
    (0.0125, 0.575), (0.0150, 0.675), (0.0168, 0.715), (0.0158, 0.642),
    (0.0151, 0.580), (0.0146, 0.512), (0.0142, 0.445), (0.0143, 0.399),
    (0.0148, 0.366), (0.0158, 0.343), (0.0184, 0.326), (0.0200, 0.330),
    (0.0250, 0.344), (0.0300, 0.372), (0.0350, 0.406), (0.0400, 0.459)]
#: Fig. (a), solid line, Simo et al., 50 % imperfection
FIG_A_SIMO_50 = [
    (0.0108, 0.411), (0.0116, 0.366), (0.0126, 0.335), (0.0141, 0.319),
    (0.0150, 0.315), (0.0200, 0.315), (0.0250, 0.344), (0.0300, 0.370),
    (0.0350, 0.401), (0.0400, 0.449)]
#: Fig. (d), solid line, Simo et al., perfect cylinder, branch switching
FIG_D_SIMO = [
    (0.0183, 0.833), (0.0182, 0.742), (0.0171, 0.695), (0.0160, 0.599),
    (0.0151, 0.530), (0.0149, 0.457), (0.0160, 0.360), (0.0190, 0.335),
    (0.0200, 0.334), (0.0250, 0.355), (0.0300, 0.397), (0.0350, 0.436),
    (0.0400, 0.480)]
#: slope of the straight line of the figures: axial deflection at load 1
FIG_LINEAR_U1 = 0.0205


def build_shell(m=12, n=12, nxy=None):
    """The ``L/10 x 15 deg`` section, loaded edge at ``x = 0``

    Edge ``x = 0`` simply supported, ``v = w = 0``, `u` free and loaded by
    the axial force per unit length ``N_CL`` (``cte=False``); edge ``x = a``
    a symmetry plane, ``u = v_x = w_x = 0``; edges ``y = 0`` and ``y = b``
    symmetry planes, ``u_y = v = w_y = 0``.

    """
    s = Shell(a=A_SEC, b=B_SEC, r=R, stack=[0.], plyt=t, laminaprop=(E, nu),
              m=m, n=n, model='cylshell_clpt_donnell')
    flags = dict(
        x1u=1, x1ur=1, x1v=0, x1vr=1, x1w=0, x1wr=1, # simple support
        x2u=0, x2ur=1, x2v=1, x2vr=0, x2w=1, x2wr=0, # symmetry
        y1u=1, y1ur=0, y1v=0, y1vr=1, y1w=1, y1wr=0, # symmetry
        y2u=1, y2ur=0, y2v=0, y2vr=1, y2w=1, y2wr=0, # symmetry
        )
    for k, v in flags.items():
        setattr(s, k, float(v))
    if nxy is not None:
        s.nx, s.ny = nxy
    s.add_distr_load_fixed_x(0., funcx=lambda y: N_CL, cte=False)
    return s


def calc_kG_N(s, N):
    r"""Geometric stiffness matrix of a membrane stress field ``N``

    `\int (N_{xx} w_{,x}^2 + 2 N_{xy} w_{,x} w_{,y} + N_{yy} w_{,y}^2) dA`
    with the stresses ``N = (Nxx, Nyy, Nxy)`` given at the integration
    points of :meth:`.Shell.integration_points`, arrays of shape ``(nx,
    ny)``.

    """
    xs, ys, wxs, wys = s.integration_points()
    m, n, a, b = s.m, s.n, s.a, s.b
    xis = 2*xs/a - 1
    etas = 2*ys/b - 1
    F = np.array([calc_vec_f(xi, s.x1w, s.x1wr, s.x2w, s.x2wr)[:m] for xi in xis])
    Fp = np.array([calc_vec_fp(xi, s.x1w, s.x1wr, s.x2w, s.x2wr)[:m] for xi in xis])*2/a
    G = np.array([calc_vec_f(eta, s.y1w, s.y1wr, s.y2w, s.y2wr)[:n] for eta in etas])
    Gp = np.array([calc_vec_fp(eta, s.y1w, s.y1wr, s.y2w, s.y2wr)[:n] for eta in etas])*2/b
    # rows: points (p, q); columns: terms j*m + i
    Bx = np.einsum('pi,qj->pqji', Fp, G).reshape(len(xs)*len(ys), m*n)
    By = np.einsum('pi,qj->pqji', F, Gp).reshape(len(xs)*len(ys), m*n)
    w = np.outer(wxs, wys).ravel()
    Nxx, Nyy, Nxy = [np.asarray(Ni).ravel()*w for Ni in N]
    kww = (Bx.T @ (Nxx[:, None]*Bx) + By.T @ (Nyy[:, None]*By)
           + Bx.T @ (Nxy[:, None]*By) + By.T @ (Nxy[:, None]*Bx))
    idx = 3*np.arange(m*n) + 2
    I, J = np.meshgrid(idx, idx, indexing='ij')
    size = s.get_size()
    return coo_matrix((kww.ravel(), (I.ravel(), J.ravel())),
                      shape=(size, size)).tocsr()


def membrane_stresses(s, c, NLgeom=True):
    """``(Nxx, Nyy, Nxy)`` of ``c`` at the integration points"""
    xs, ys = s.integration_points()[:2]
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    fields = s.stress(c, xs=X, ys=Y, NLgeom=NLgeom)[1]
    return fields['Nxx'], fields['Nyy'], fields['Nxy']


def strain_energy_imp(s, c, c0):
    """Strain energy of the imperfect shell, from the generalized strains
    ``eps(c + c0) - eps(c0)`` at the integration points"""
    xs, ys, wxs, wys = s.integration_points()
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    names = s._strain_names()
    e1 = s.strain(c + c0, xs=X, ys=Y, NLgeom=True)[1]
    e1 = np.array([e1[k] for k in names])
    e0 = s.strain(c0, xs=X, ys=Y, NLgeom=True)[1]
    e0 = np.array([e0[k] for k in names])
    e = (e1 - e0).reshape(len(names), -1)
    w = np.outer(wxs, wys).ravel()
    return 0.5*np.einsum('ip,ij,jp,p->', e, s.ABD, e, w)


class ImperfectShell(object):
    r"""Callables of structsolve for the shell with an initial imperfection

    For the Donnell kinematics the membrane strains and the curvatures of
    the imperfect shell are ``eps(c + c0) - eps(c0)``, which gives

    ``fint_imp(c) = fint(c + c0) - fint(c0) - kG(N0) c``

    ``kT_imp(c) = kT(c + c0) - kG(N0)``

    with ``N0`` the membrane resultants, linear and non-linear parts, of the
    state ``c0``, see the documentation page.

    """
    def __init__(self, s, c0):
        self.s = s
        self.c0 = np.asarray(c0, dtype=float)
        self.fint0 = s.calc_fint(self.c0)
        self.kG0 = calc_kG_N(s, membrane_stresses(s, self.c0, NLgeom=True))
        self.kT0 = self.calc_kT(np.zeros_like(self.c0))
        self.size = s.get_size()

    def calc_fext(self, inc=1., silent=True):
        return self.s.calc_fext(inc=inc, silent=silent)

    def calc_fint(self, c, silent=True):
        return self.s.calc_fint(c + self.c0) - self.fint0 - self.kG0 @ c

    def calc_kT(self, c):
        return csr_matrix(self.s.calc_kT(c=c + self.c0) - self.kG0)

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        if c is None or not NLgeom:
            return self.kT0
        return self.calc_kT(c)

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        return csr_matrix((self.size, self.size))


class ScaledModel(object):
    r"""The callables of a model in the scaled unknowns ``c = S c_s``

    ``S = diag(1/sqrt(K_ii))``, with the diagonal of the linear stiffness
    matrix ``K = calc_kC()``, and ``S_ii = 1`` on its null rows, such that
    the scaled linear stiffness matrix has a unit diagonal. The structural
    problem is unchanged, only the coordinates of the Ritz vector are: the
    force vectors become ``S f`` and the matrices ``S K S``. The Ritz
    constants of Bardell's functions mix values and derivatives of very
    different magnitudes; in the scaled unknowns the eigenvalues of the
    tangent stiffness matrix closest to zero, and hence the inertia of
    PARDISO and the critical eigenvector of the branch switching, are those
    of the physical instabilities rather than of weakly stiff Ritz terms.

    """
    def __init__(self, calc_fext, calc_fint, calc_kC, calc_kG):
        self._fext, self._fint = calc_fext, calc_fint
        self._kC, self._kG = calc_kC, calc_kG
        d = self._kC().diagonal()
        self.S = np.ones_like(d)
        self.S[d > 0] = 1/np.sqrt(d[d > 0])
        self._Sm = diags(self.S)

    def unscale(self, cs):
        return self.S*cs

    def calc_fext(self, inc=1., silent=True):
        return self.S*self._fext(inc=inc, silent=silent)

    def calc_fint(self, c, silent=True):
        return self.S*self._fint(self.S*c, silent=silent)

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        c = None if c is None else self.S*c
        return csr_matrix(self._Sm @ self._kC(c=c, NLgeom=NLgeom, silent=silent) @ self._Sm)

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        c = None if c is None else self.S*c
        return csr_matrix(self._Sm @ self._kG(c=c, NLgeom=NLgeom, silent=silent) @ self._Sm)


def edge_shortening(s, c, ny=61):
    """Average and maximum of ``u`` at the loaded edge ``x = 0``"""
    ys = np.linspace(0, s.b, ny)
    u = s.uvw(c, xs=np.zeros_like(ys), ys=ys)[1]['u']
    u = np.ravel(u)
    avg = np.trapezoid(u, ys)/s.b
    return avg, u.max()


def linear_buckling(s, num_eigvalues=5, silent=True):
    """Linear buckling load factor of the reference load and modes"""
    fext = s.calc_fext()
    K = s.calc_kC()
    clin = solve(K, fext, silent=True)
    KG = s.calc_kG(c=clin)
    eigvals, eigvecs = lb(K, KG, num_eigvalues=num_eigvalues, silent=silent)
    return eigvals, eigvecs, clin


def imperfection(s, mode, xi):
    """Ritz vector of the imperfection: the `w` part of ``mode`` scaled to
    ``max|w0| = xi*t``"""
    c0 = np.zeros_like(mode)
    c0[2::3] = mode[2::3]
    w = s.uvw(c0, gridx=101, gridy=101)[1]['w']
    imax = np.argmax(np.abs(w))
    return c0*xi*t/np.ravel(w)[imax]


def normalized_shell(m, n, nxy=None):
    """Shell whose reference load is the linear buckling load"""
    s = build_shell(m, n, nxy)
    eigvals, eigvecs, _ = linear_buckling(s)
    lbd_cr = eigvals[0]
    s.distr_loads_inc = []
    s.add_distr_load_fixed_x(0., funcx=lambda y: lbd_cr*N_CL, cte=False)
    return s, lbd_cr, np.real(eigvecs[:, 0])


def run_riks(xi=0.1, m=12, n=12, umax=0.04, branch_switching=0, silent=False,
             scaled=None, save=True, verbose=True, **kwargs):
    """Riks analysis until the maximum shortening ``umax``

    ``xi > 0``: imperfection of amplitude ``xi*t`` in the shape of the
    lowest linear buckling mode, see :class:`ImperfectShell`; ``xi = 0``:
    the perfect shell, with ``inertia`` (PARDISO) and ``branch_switching``.
    ``scaled`` (default: ``True`` for the perfect shell) solves in the
    scaled unknowns of :class:`ScaledModel`. ``kwargs`` are attributes of
    the :class:`structsolve.Analysis`. ``save`` writes the ``.npz`` file
    read by :func:`summary` and :func:`plot`.

    Returns a dictionary: the analysis ``an``, the shell ``s``, the linear
    buckling load ``lbd_cr`` (``/N_CL``), the load factors ``lbds``, the
    average and maximum axial deflections ``u`` (one row per increment), the
    maximum deflections ``wmax``, the Ritz vectors ``cs`` (unscaled), the
    indices ``branch_points`` of the switched states, the run time
    ``elapsed`` and the maximum relative residual ``max_residual``.

    """
    log = print if verbose else (lambda *args: None)
    s, lbd_cr, mode = normalized_shell(m, n)
    log('m=%d n=%d linear buckling N/N_CL = %.5f' % (m, n, lbd_cr))
    if xi > 0:
        c0 = imperfection(s, mode, xi)
        model = ImperfectShell(s, c0)
        funcs = (model.calc_fext, model.calc_fint, model.calc_kC, model.calc_kG)
    else:
        c0 = np.zeros(s.get_size())
        funcs = (s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    if scaled is None:
        scaled = xi == 0
    if scaled:
        model = ScaledModel(*funcs)
        funcs = (model.calc_fext, model.calc_fint, model.calc_kC, model.calc_kG)
        unscale = model.unscale
    else:
        unscale = lambda c: c
    an = Analysis(*funcs)
    an.NL_method = 'arc_length_riks'
    an.final_load_factor = None
    an.initialInc = 0.1
    an.maxInc = 0.1
    if xi == 0:
        an.branch_switching = branch_switching
        an.inertia = True
        # the sign of the determinant of SuperLU, as for the figure of the
        # documentation; with linear_solver='pardiso' (keyword argument)
        # the count is exact since the fallback of static._factorize for
        # perturbed pivots, and the switch works also with 10 x 10 and
        # 12 x 12 terms
        an.linear_solver = 'splu'
        an.initialInc = an.maxInc = 0.03
    an.minInc = 1e-5
    an.maxArcLength = 1e4
    an.stop_criterion = value_reached(
        lambda c: edge_shortening(s, unscale(c))[1], umax)
    for k, v in kwargs.items():
        setattr(an, k, v)
    t0 = time.perf_counter()
    an.static(NLgeom=True, silent=silent)
    elapsed = time.perf_counter() - t0
    lbds = np.asarray(an.increments)
    # equilibrium of every converged increment
    fint = an.calc_fint
    fext = an.calc_fext()
    res = max(np.linalg.norm(fint(c) - lbd*fext)/np.linalg.norm(fext)
              for lbd, c in zip(lbds, an.cs))
    cs = [unscale(c) for c in an.cs]
    u = np.array([edge_shortening(s, c) for c in cs])
    wmax = np.array([np.abs(s.uvw(c, gridx=21, gridy=21)[1]['w']).max()
                     for c in cs])
    log('increments %d, time %.1f s, max relative residual %.1e'
        % (len(lbds), elapsed, res))
    ilim = np.argmax(np.diff(lbds) < 0) if np.any(np.diff(lbds) < 0) else -1
    if ilim >= 0:
        log('first limit load %.4f at u_avg %.5f u_max %.5f'
            % (lbds[ilim], u[ilim, 0], u[ilim, 1]))
    if save:
        case = 'xi%g' % xi if xi > 0 else 'perfect'
        fname = os.path.join(OUT, 'simo1990_%s_m%dn%d.npz' % (case, m, n))
        np.savez(fname, lbds=lbds, u=u, wmax=wmax, lbd_cr=lbd_cr,
                 elapsed=elapsed, cs=np.asarray(cs), c0=c0,
                 branch_points=np.asarray(an.branch_points, dtype=int),
                 num_negative_eigvals=np.asarray(
                     [-1 if v is None else v for v in an.num_negative_eigvals]))
        log('saved', fname)
    return dict(an=an, s=s, lbd_cr=lbd_cr, lbds=lbds, u=u, wmax=wmax, cs=cs,
                branch_points=np.asarray(an.branch_points, dtype=int),
                elapsed=elapsed, max_residual=res)


def path_values(lbds, u_max):
    """Characteristic values of a computed path

    ``lbds`` and ``u_max`` of the converged increments. Returns a dictionary
    with the first limit (or bifurcation) load ``P_lim`` and ``u_lim`` at
    it, the smallest deflection after it ``u_back`` at the load ``P_back``
    (the snap-back), the minimum load after it ``P_min`` at ``u_min``, and
    the loads ``P(0.02)``, ``P(0.025)``, ``P(0.03)``, ``P(0.035)`` and
    ``P(0.04)`` at these maximum deflections on the post-buckled branch,
    after the snap-back (``nan`` beyond the end of the path).

    """
    lbds = np.asarray(lbds)
    u_max = np.asarray(u_max)
    ilim = np.flatnonzero(np.diff(lbds) < 0)[0]
    imin = ilim + np.argmin(lbds[ilim:])
    iback = ilim + np.argmin(u_max[ilim:imin + 1])
    out = dict(P_lim=lbds[ilim], u_lim=u_max[ilim], P_min=lbds[imin],
               u_min=u_max[imin], u_back=u_max[iback], P_back=lbds[iback])
    for ui in (0.02, 0.025, 0.03, 0.035, 0.04):
        out['P(%g)' % ui] = (np.interp(ui, u_max[iback:], lbds[iback:])
                             if u_max[-1] >= ui else np.nan)
    return out


def verify(m=8, n=8, xi=0.5, seed=1):
    """Checks of the imperfect model, printed"""
    rng = np.random.default_rng(seed)
    s, lbd_cr, mode = normalized_shell(m, n)
    size = s.get_size()
    # 1) calc_kG_N with the linear stresses equals panels' calc_kG(c)
    c = rng.standard_normal(size)*1e-2
    kG_p = s.calc_kG(c=c, NLgeom=True).toarray()
    kG_m = calc_kG_N(s, membrane_stresses(s, c, NLgeom=False)).toarray()
    print('kG(N_L(c)), this script vs. panels calc_kG(c): rel. diff %.2e'
          % (np.abs(kG_m - kG_p).max()/np.abs(kG_p).max()))
    # 2) panels' calc_kG(c) does not contain the non-linear stresses
    kG_full = calc_kG_N(s, membrane_stresses(s, c, NLgeom=True)).toarray()
    print('kG(N(c)) vs. calc_kG(c): rel. diff %.2e (non-linear stresses '
          'missing in calc_kG)' % (np.abs(kG_full - kG_p).max()/np.abs(kG_p).max()))
    # 3) imperfect model
    c0 = imperfection(s, mode, xi)
    model = ImperfectShell(s, c0)
    print('|fint_imp(0)| / |fint(c0)| = %.2e'
          % (np.linalg.norm(model.calc_fint(0*c0))/np.linalg.norm(model.fint0)))
    c = np.zeros(size)
    c[2::3] = rng.standard_normal(size//3)
    c *= 0.3*t/np.abs(s.uvw(c, gridx=31, gridy=31)[1]['w']).max()
    c[0::3] = rng.standard_normal(size//3)*1e-3
    c[1::3] = rng.standard_normal(size//3)*1e-3
    v = rng.standard_normal(size)
    v[(np.abs(model.kT0).sum(axis=1).A.ravel() == 0)] = 0
    kT = model.calc_kT(c)
    for h in (1e-4, 1e-5, 1e-6):
        dfd = (model.calc_fint(c + h*v) - model.calc_fint(c - h*v))/(2*h)
        print('h=%.0e |(f(c+hv)-f(c-hv))/2h - kT_imp v|/|kT_imp v| = %.2e'
              % (h, np.linalg.norm(dfd - kT @ v)/np.linalg.norm(kT @ v)))
    # 4) independent check of fint_imp: gradient of the strain energy of the
    #    imperfect shell, integrated from the strains eps(c + c0) - eps(c0)
    for h in (1e-4, 1e-5):
        dU = (strain_energy_imp(s, c + h*v, c0)
              - strain_energy_imp(s, c - h*v, c0))/(2*h)
        fv = model.calc_fint(c) @ v
        print('h=%.0e (U(c+hv)-U(c-hv))/2h vs fint_imp(c).v: rel. diff %.2e'
              % (h, abs(dU - fv)/abs(fv)))
    # ... which fails with panels' calc_kG(c0), of the linear stresses only
    kG0_L = s.calc_kG(c=c0, NLgeom=True)
    fv = (s.calc_fint(c + c0) - model.fint0 - kG0_L @ c) @ v
    print('same with calc_kG(c0) in place of kG(N0): rel. diff %.2e'
          % (abs(dU - fv)/abs(fv)))
    # 5) without imperfection the model is the perfect one
    model0 = ImperfectShell(s, 0*c0)
    print('xi=0: |fint_imp(c) - fint(c)|/|fint(c)| = %.2e'
          % (np.linalg.norm(model0.calc_fint(c) - s.calc_fint(c))
             /np.linalg.norm(s.calc_fint(c))))


def _load(case, m, n):
    fname = os.path.join(OUT, 'simo1990_%s_m%dn%d.npz' % (case, m, n))
    if not os.path.isfile(fname):
        return None
    d = np.load(fname)
    lbds = np.concatenate(([0.], d['lbds']))
    u = np.vstack(([0., 0.], d['u']))
    return lbds, u, d


def plot(fname, m=12, n=12, mconv=(8, 10, 14), m_perfect=8):
    """Computed curves over the values read from Fig. 6.3.4"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    ulin = np.array([0, 0.04])
    for ax in axes:
        ax.plot(ulin, ulin/FIG_LINEAR_U1, color='0.6', lw=0.8)
    ax = axes[0]
    colors = {'xi0.1': 'C0', 'xi0.5': 'C3'}
    labels = {'xi0.1': '10 %', 'xi0.5': '50 %'}
    for case in ('xi0.1', 'xi0.5'):
        for mm in mconv:
            r = _load(case, mm, mm)
            if r is not None:
                ax.plot(r[1][:, 1], r[0], color=colors[case], lw=0.6,
                        ls=':')
        r = _load(case, m, n)
        if r is not None:
            ax.plot(r[1][:, 1], r[0], color=colors[case], lw=1.5,
                    label='structsolve, %s, maximum' % labels[case])
            ax.plot(r[1][:, 0], r[0], color=colors[case], lw=0.8, ls='--',
                    label='structsolve, %s, average' % labels[case])
    for data, mk, lab in ((FIG_A_SIMO_10, 'o', 'Simo et al., Fig. (a), 10 %'),
                          (FIG_A_SIMO_50, 's', 'Simo et al., Fig. (a), 50 %'),
                          (FIG_A_ANS_10, '^', 'Stanley, Fig. (a), 10 %'),
                          (FIG_A_ANS_50, 'D', 'Stanley, Fig. (a), 50 %')):
        d = np.array(data)
        ax.plot(d[:, 0], d[:, 1], mk, ms=4, mfc='none', color='k', mew=0.7,
                label=lab)
    ax.set_title('Imperfection in the lowest linear buckling mode')
    ax.set_ylabel('Axial load / linear buckling load')
    ax = axes[1]
    r = _load('perfect', m_perfect, m_perfect)
    if r is not None:
        ax.plot(r[1][:, 1], r[0], color='C2', lw=1.5,
                label='structsolve, m = n = %d, branch switching' % m_perfect)
        bp = r[2]['branch_points'] + 1
        ax.plot(r[1][bp, 1], r[0][bp], 'x', color='C2', ms=7,
                label='switched to a bifurcated branch')
    d = np.array(FIG_D_SIMO)
    ax.plot(d[:, 0], d[:, 1], 'o', ms=4, mfc='none', color='k', mew=0.7,
            label='Simo et al., Fig. (d)')
    ax.set_title('Perfect cylinder')
    for ax in axes:
        ax.set_xlim(0, 0.042)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Axial deflection of the loaded edge')
        ax.grid(lw=0.3)
        ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    print('saved', fname)


def summary(cases=('xi0.1', 'xi0.5', 'perfect'), mns=(8, 10, 12, 14)):
    """Table of the computed limit loads and post-buckling values"""
    print('case     m=n  lbd_cr  P_lim  u_max  u_avg  P_min  u_max  '
          'P(0.04) incs  time')
    for case in cases:
        for mm in mns:
            r = _load(case, mm, mm)
            if r is None:
                continue
            lbds, u, d = r
            drops = np.flatnonzero(np.diff(lbds) < 0)
            if drops.size == 0:
                print('%-8s %3d  %.4f  no limit point, max. load %.3f'
                      % (case, mm, d['lbd_cr'], lbds.max()))
                continue
            ilim = drops[0]
            imin = ilim + np.argmin(lbds[ilim:])
            i1 = np.flatnonzero(u[:, 1] >= 0.04)
            p04 = np.nan
            if i1.size:
                i1 = i1[0]
                p04 = np.interp(0.04, u[i1-1:i1+1, 1], lbds[i1-1:i1+1])
            print('%-8s %3d  %.4f  %.3f  %.4f %.4f  %.3f  %.4f  %.3f  %4d %6.0f'
                  % (case, mm, d['lbd_cr'], lbds[ilim], u[ilim, 1],
                     u[ilim, 0], lbds[imin], u[imin, 1], p04,
                     len(lbds) - 1, d['elapsed']))


def main(argv):
    cmd = argv[1] if len(argv) > 1 else 'verify'
    if cmd == 'verify':
        verify()
    elif cmd == 'lb':
        m, n = int(argv[2]), int(argv[3])
        s = build_shell(m, n)
        eigvals, eigvecs, clin = linear_buckling(s, silent=True)
        print('m=%d n=%d eigenvalues N/N_CL:' % (m, n), eigvals[:5])
        u = edge_shortening(s, clin*eigvals[0])
        print('shortening at the linear buckling load: avg %.5f max %.5f'
              % u)
    elif cmd == 'riks':
        xi, m, n = float(argv[2]), int(argv[3]), int(argv[4])
        run_riks(xi, m, n, silent=True)
    elif cmd == 'perfect':
        m, n = int(argv[2]), int(argv[3])
        nbs = int(argv[4]) if len(argv) > 4 else 3
        kwargs = {'linear_solver': argv[5]} if len(argv) > 5 else {}
        run_riks(0., m, n, branch_switching=nbs, silent=True, **kwargs)
    elif cmd == 'summary':
        summary()
    elif cmd == 'plot':
        plot(argv[2] if len(argv) > 2 else 'ex_cylinder_postbuckling.png')
    else:
        raise ValueError(cmd)


if __name__ == '__main__':
    main(sys.argv)
