r"""Interface to the sparse direct solver PARDISO of Intel oneMKL

Used by ``Analysis.linear_solver = 'pardiso'``, see :func:`.static._factorize`.
The shared library ``mkl_rt`` of oneMKL is given by the environment variable
``STRUCTSOLVE_MKL_RT``, otherwise found by :func:`ctypes.util.find_library`
or under ``sys.prefix`` or the user site, as installed by ``pip install mkl``
(the optional dependency ``structsolve[pardiso]``) or by conda.

:class:`PardisoSolver` calls the phases of PARDISO separately, in particular
the phase 22, a numerical factorization that reuses the analysis (ordering
and symbolic factorization) of the last phase 12 for a matrix of the same
pattern.

"""
# The location of mkl_rt and the call of pardiso are adapted from pypardiso
# 0.4.7, https://github.com/haasad/PyPardiso, under the license:
#
# Copyright (c) 2016, Adrian Haas and ETH Zürich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer. Redistributions in
# binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other
# materials provided with the distribution.
# Neither the name of ETH Zürich nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import ctypes
import glob
import os
import site
import sys
from ctypes.util import find_library

import numpy as np

#: Environment variable with the path of the shared library ``mkl_rt``
MKL_RT_ENV = 'STRUCTSOLVE_MKL_RT'

_mkl = None


def load_mkl():
    """The function ``pardiso`` of the shared library ``mkl_rt``, loaded
    once; raises :class:`ImportError` when it is not found"""
    global _mkl
    if _mkl is not None:
        return _mkl
    path = os.environ.get(MKL_RT_ENV)
    if path is None:
        path = find_library('mkl_rt') or find_library('mkl_rt.1')
    if path is not None:
        candidates = [path]
    else:
        # lib (Linux and macOS) or Library/bin (Windows) of the environment,
        # where pip and conda install mkl, otherwise any subdirectory of the
        # environment or of the user site, shorter paths first
        candidates = (
            glob.glob('%s/lib/libmkl_rt.*' % sys.prefix)
            + glob.glob('%s/Library/bin/mkl_rt*.dll' % sys.prefix)
            or sorted(glob.glob('%s/[Ll]ib*/**/*mkl_rt*' % sys.prefix,
                                recursive=True)
                      or glob.glob('%s/[Ll]ib*/**/*mkl_rt*' % site.USER_BASE,
                                   recursive=True), key=len))
    for p in candidates:
        try:
            lib = ctypes.CDLL(p)
            f = lib.pardiso
            break
        except (OSError, AttributeError):
            pass
    else:
        raise ImportError("linear_solver='pardiso' requires the shared "
                          "library mkl_rt of Intel oneMKL, e.g. pip install "
                          "mkl; its path may be given by the environment "
                          "variable %s" % MKL_RT_ENV)
    i32 = ctypes.POINTER(ctypes.c_int32)
    f.argtypes = [ctypes.POINTER(ctypes.c_int64),  # pt
                  i32, i32, i32, i32, i32,  # maxfct, mnum, mtype, phase, n
                  ctypes.c_void_p, i32, i32,  # a, ia, ja
                  i32, i32, i32, i32,  # perm, nrhs, iparm, msglvl
                  ctypes.c_void_p, ctypes.c_void_p,  # b, x
                  i32]  # error
    f.restype = None
    _mkl = f
    return _mkl


def available():
    """Whether PARDISO can be used, i.e. ``mkl_rt`` is found"""
    try:
        load_mkl()
    except ImportError:
        return False
    return True


class PardisoError(RuntimeError):
    """Error code of PARDISO, ``error`` not 0"""
    def __init__(self, phase, error):
        self.phase = phase
        self.error = error
        super().__init__('PARDISO failed in the phase %d with the error %d'
                         % (phase, error))


class PardisoSolver(object):
    r"""PARDISO solver of one real matrix, by default symmetric indefinite
    (``mtype = -2``), given by its upper triangle in CSR format with every
    diagonal term stored

    ``iparm`` is the array of the parameters of PARDISO, indexed from 0,
    i.e. ``iparm[i - 1]`` is ``iparm(i)`` of the manual of oneMKL: all
    zeros, the default parameters, which PARDISO fills, or set by the user
    with ``iparm[0] = 1``. After a factorization, ``iparm[13]`` is the
    number of perturbed pivots and ``iparm[21]`` and ``iparm[22]`` the
    numbers of positive and negative eigenvalues (``mtype = -2``).

    The matrix of the last factorization is kept, not copied, for the
    iterative refinement of :meth:`solve`, and must not be changed.

    """
    def __init__(self, mtype=-2):
        self._pardiso = load_mkl()
        self.mtype = mtype
        self.pt = np.zeros(64, dtype=np.int64)
        self.iparm = np.zeros(64, dtype=np.int32)
        self.n = 0
        # 1-based CSR indices of the last analysis and values of the last
        # factorization
        self._ia = None
        self._ja = None
        self._a = None
        self.factorized = False

    @property
    def analyzed(self):
        """Whether the solver holds an analysis, for the phase 22"""
        return self._ia is not None

    def _call(self, phase, b=None):
        nrhs = 1 if b is None or b.ndim == 1 else b.shape[1]
        if b is None:
            b = np.zeros(max(self.n, 1))
        x = np.zeros_like(b)
        dummy = np.zeros(1, dtype=np.int32)
        ia = self._ia if self._ia is not None else dummy
        ja = self._ja if self._ja is not None else dummy
        a = self._a if self._a is not None else np.zeros(1)
        i32 = ctypes.POINTER(ctypes.c_int32)
        error = ctypes.c_int32(0)
        self._pardiso(self.pt.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                      ctypes.byref(ctypes.c_int32(1)),
                      ctypes.byref(ctypes.c_int32(1)),
                      ctypes.byref(ctypes.c_int32(self.mtype)),
                      ctypes.byref(ctypes.c_int32(phase)),
                      ctypes.byref(ctypes.c_int32(self.n)),
                      a.ctypes.data, ia.ctypes.data_as(i32),
                      ja.ctypes.data_as(i32), dummy.ctypes.data_as(i32),
                      ctypes.byref(ctypes.c_int32(nrhs)),
                      self.iparm.ctypes.data_as(i32),
                      ctypes.byref(ctypes.c_int32(0)),
                      b.ctypes.data, x.ctypes.data, ctypes.byref(error))
        if error.value != 0:
            raise PardisoError(phase, error.value)
        return x

    def factorize(self, A, phase=12):
        """Factorization of the CSR matrix ``A``, ``float64``, with the
        phase 12 (analysis and numerical factorization) or
        22 (numerical factorization with the analysis of the last phase 12,
        ``A`` having its pattern)"""
        n = A.shape[0]
        if A.shape[1] != n:
            raise ValueError('PardisoSolver: the matrix is not square')
        if A.dtype != np.float64:
            raise ValueError('PardisoSolver: float64 matrix expected')
        if not A.has_sorted_indices:
            A = A.sorted_indices()
        if not np.all(np.diff(A.indptr)):
            # an empty row makes PARDISO crash
            raise ValueError('PardisoSolver: the matrix has an empty row')
        if phase == 12:
            self.n = n
            self._ia = A.indptr.astype(np.int32) + 1
            self._ja = A.indices.astype(np.int32) + 1
        elif phase == 22:
            if not self.analyzed or n != self.n or A.nnz != self._ja.shape[0]:
                raise ValueError('PardisoSolver: the phase 22 needs the '
                                 'analysis of a matrix of the same pattern')
        else:
            raise ValueError('PardisoSolver: phase 12 or 22')
        self.factorized = False
        self._a = np.ascontiguousarray(A.data)
        self._call(phase)
        self.factorized = True

    def solve(self, b):
        """Solution of the last factorization (phase 33), ``b`` of shape
        ``(n,)`` or ``(n, nrhs)``"""
        if not self.factorized:
            raise ValueError('PardisoSolver: no factorization')
        b = np.asarray(b, dtype=np.float64)
        if b.shape[0] != self.n:
            raise ValueError('PardisoSolver: b of %d rows instead of %d'
                             % (b.shape[0], self.n))
        x = self._call(33, np.asfortranarray(b))
        return np.ascontiguousarray(x)

    def free(self, everything=False):
        """Frees the memory of the factorization (phase 0), keeping the
        analysis for the phase 22, or all the memory of PARDISO (phase
        -1)"""
        if self.analyzed or self.factorized:
            self._call(-1 if everything else 0)
        self.factorized = False
        self._a = None
        if everything:
            self._ia = None
            self._ja = None
