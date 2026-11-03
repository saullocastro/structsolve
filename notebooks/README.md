# Notebooks

Validation notebooks of `structsolve` against results of the literature and
the utilities they share. They are run from this folder. The structural
models come from [`panels`](https://github.com/saullocastro/panels), which
requires `structsolve>=0.10.0` and is not a dependency of `structsolve`:
install it (`pip install panels`, or its repository built in place with
`python setup.py build_ext --inplace` and on the `PYTHONPATH`) to run them,
with oneMKL (`pip install mkl`) for `linear_solver='pardiso'`.

The tests of `tests/` that use these utilities import them from this folder
and are skipped where `panels` is not installed.

| File | Content |
|---|---|
| `simo1990_cylinder_postbuckling.ipynb` | Simo, Fox and Rifai (1990): post-buckling of an axially compressed cylinder with imperfections of 10 % and 50 % of the thickness, Riks method, and of the perfect one by branch switching with `linear_solver='pardiso'` and `'splu'`; runs `panels` and `structsolve` at each execution (about 3.5 min on one thread), with `simo1990_utils.py`. Tested in `tests/test_simo1990_cylinder.py`, documentation page `ex_cylinder_postbuckling` |
| `simo1990_utils.py` | Model of the cylinder (`panels` shell, exact Donnell imperfection `ImperfectShell`, scaled unknowns `ScaledModel`), Riks analyses (`run_riks`), characteristic values of a path (`path_values`), values read from Fig. 6.3.4, and the command line of the convergence runs of the documentation page (`verify`, `lb`, `riks`, `perfect`, `summary`, `plot`, see its docstring) |
