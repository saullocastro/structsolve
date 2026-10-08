// Run the test suite of structsolve in Pyodide (WebAssembly), with Node.js
//
// usage: node run_tests.mjs <wheel dir> <tests dir> [pytest args...]
//
// The wheel of structsolve found in <wheel dir> is installed with micropip,
// NumPy, SciPy and pytest are the packages distributed with Pyodide.
import { loadPyodide } from "pyodide";

const [wheelDir, testsDir, ...args] = process.argv.slice(2);
const pyodide = await loadPyodide();
await pyodide.loadPackage(["micropip", "numpy", "scipy", "pytest"]);
for (const [mount, root] of [["/dist", wheelDir], ["/tests", testsDir]]) {
  pyodide.FS.mkdir(mount);
  pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root }, mount);
}
pyodide.globals.set("ARGS", pyodide.toPy(args));
const code = await pyodide.runPythonAsync(`
import glob
import sys

import micropip

wheels = glob.glob("/dist/structsolve-*.whl")
assert len(wheels) == 1, wheels
await micropip.install("emfs:" + wheels[0])

import numpy
import pytest
import scipy
import structsolve

print("Python", sys.version.split()[0], sys.platform, "NumPy", numpy.__version__,
      "SciPy", scipy.__version__, "structsolve", structsolve.__file__)
assert "site-packages" in structsolve.__file__
int(pytest.main(["/tests", "-p", "no:cacheprovider", *ARGS]))
`);
process.exit(code);
