r"""Pinned column discretized with finite differences

A column of length ``L`` with ``N`` segments of length ``h = L/N`` and the
nodes ``0, ..., N``. Each node has an axial displacement ``u_i`` and a
transverse displacement ``w_i``, ordered as ``[u_0, ..., u_N, w_0, ...,
w_N]``. The strain energy is

.. math::

    U = \sum_{e=0}^{N-1} \frac{EA}{2h} (u_{e+1} - u_e)^2
      + \sum_{i=1}^{N-1} \frac{EI}{2h^3} (w_{i-1} - 2 w_i + w_{i+1})^2

and the geometric stiffness matrix of the axial forces `N_e = EA (u_{e+1}
- u_e)/h`, positive in tension, comes from `\sum_e \frac{N_e}{2h} (w_{e+1}
- w_e)^2`. The masses are lumped at the nodes.

With ``u_0 = w_0 = w_N = 0`` and a compressive load ``P`` at ``u_N``, the
load multipliers and the natural frequencies of the transverse modes are,
with `\ell_k = 4 \sin^2(k \pi h/(2L))`:

.. math::

    \lambda_k P = \frac{EI \ell_k}{h^2}, \qquad
    \omega_k^2 = \frac{EI \ell_k^2}{m h^4}

which tend to the Euler loads `k^2 \pi^2 EI/L^2` and to the frequencies
`(k \pi/L)^2 \sqrt{EI/m}` of the pinned column when ``N`` grows.

"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags


class Column(object):
    """Matrices of the pinned column, see the module docstring"""
    def __init__(self, N=20, L=1., EI=1., EA=1.e5, m=1.):
        self.N = N
        self.L = L
        self.EI = EI
        self.EA = EA
        self.m = m
        self.h = L/N
        self.size = 2*(N + 1)
        #: constrained dofs: u_0, w_0 and w_N
        self.bc = np.array([self.u(0), self.w(0), self.w(N)])
        #: dof of the load
        self.tip = self.u(N)

    def u(self, i):
        return i

    def w(self, i):
        return self.N + 1 + i

    def _assemble(self, terms):
        rows, cols, vals = [], [], []
        for k, dofs, coeffs in terms:
            for a, ca in zip(dofs, coeffs):
                for b, cb in zip(dofs, coeffs):
                    rows.append(a)
                    cols.append(b)
                    vals.append(k*ca*cb)
        return csr_matrix(coo_matrix((vals, (rows, cols)),
                                     shape=(self.size, self.size)))

    def K(self):
        """Linear stiffness matrix, without boundary conditions"""
        N, h = self.N, self.h
        terms = [(self.EA/h, [self.u(e), self.u(e+1)], [-1., 1.])
                 for e in range(N)]
        terms += [(self.EI/h**3, [self.w(i-1), self.w(i), self.w(i+1)],
                   [1., -2., 1.]) for i in range(1, N)]
        return self._assemble(terms)

    def KG(self, c):
        """Geometric stiffness matrix of the axial forces of ``c``"""
        N, h = self.N, self.h
        terms = []
        for e in range(N):
            Ne = self.EA*(c[self.u(e+1)] - c[self.u(e)])/h
            terms.append((Ne/h, [self.w(e), self.w(e+1)], [-1., 1.]))
        return self._assemble(terms)

    def M(self):
        """Lumped mass matrix"""
        mass = self.m*self.h*np.ones(self.N + 1)
        mass[0] = mass[-1] = self.m*self.h/2
        return csr_matrix(diags(np.concatenate((mass, mass))))

    def fext(self, P=1.):
        """Compressive load ``P`` at the end ``u_N``"""
        f = np.zeros(self.size)
        f[self.tip] = -P
        return f

    def ell(self, k):
        return 4*np.sin(k*np.pi*self.h/(2*self.L))**2

    def load_multipliers(self, k, P=1.):
        """Exact ``lambda_k`` of the discrete problem for the load ``P``"""
        return self.EI*self.ell(k)/(self.h**2*P)

    def omega2(self, k):
        """Exact ``omega_k**2`` of the discrete transverse modes"""
        return self.EI*self.ell(k)**2/(self.m*self.h**4)


def apply_null_rows(A, dofs):
    """Zero the rows and columns ``dofs`` of the sparse matrix ``A``"""
    keep = np.ones(A.shape[0])
    keep[dofs] = 0.
    D = diags(keep)
    A = csr_matrix(D @ A @ D)
    A.eliminate_zeros()
    return A
