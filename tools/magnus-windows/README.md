# pymc + magnus.NSFeas on Windows: pinned, patched, reproducible

The notebooks in `cstr-cyclic/` use the nested-sampling feasibility solver `NSFeas` of
MAGNUS on top of MC++ (`import pymc`, `from magnus import NSFeas`). Upstream builds these
with GNU make on Linux only. This folder builds them natively on Windows with the
Microsoft compiler, from sources pinned to exact commits, with one patch to MAGNUS
(two NSFeas bugs, see below), and puts the result into the venv of your choice.

Contents:

- `setup.py`: does everything except the venv management (clone, patch, download, build,
  optionally install, test). Standard library only.
- `CMakeLists.txt`, `win/magnus_nsfeas_only.cpp`: the Windows build.
- `patches/0001-nsfeas-failed-proposal-alignment-and-maxcpu.patch`: the MAGNUS fix.
- `tests/`: verification scripts (`smoke.py`, `failed_eval_alignment.py`, `maxcpu.py`,
  `exception_in_callback.py`).

## Quick start

Requirements, once per machine: git; Visual Studio Build Tools with the workload
"Desktop development with C++" (MSVC, Windows SDK, CMake, Ninja); a CPython 3.12+ x64
install that ships its C headers (the Python install manager and python.org installers do).

From the repository root, after `uv sync` created `.venv`:

```
.venv\Scripts\python.exe tools\magnus-windows\setup.py --install
```

Run it with the Python of the venv that should receive the modules. The modules are built
against that interpreter's base CPython and installed into that venv's site-packages as
exactly three files: `pymc.cp3XX-win_amd64.pyd`, `magnus.cp3XX-win_amd64.pyd`,
`libopenblas.dll`. Restart the Jupyter kernel afterwards.

Any prompt works; the script locates the Visual Studio tools with `vswhere`. First run:
about 340 MB of downloads and 3 to 5 minutes of compiling. Clones, downloads and the
build live in `%LOCALAPPDATA%\magnus-windows-build` (change with `--work-dir`), never in
the repository. Reruns skip everything that is already done.

Options: `--install` (without it the script only builds and prints the install command),
`--work-dir DIR`, `--jobs N` (default 4), `--skip-tests`, `--allow-base-python`.

To remove the modules, delete the three files from the venv's site-packages. They carry
no pip metadata, so `uv pip uninstall` does not know them. `uv sync` leaves them alone;
recreating the venv removes them, so rerun the command above.

## Pinned versions

| Component | Source | Pinned to |
|---|---|---|
| MC++ 4.0 | github.com/omega-icl/mcpp, branch version-4.0 | commit 7b70a95535fb49cffbab7890e63accf7c4a7b5b1 (2025-11-21) |
| pybind11 | submodule of mcpp | commit eef24d6172f4a46bcc68f91d3e1ceba71cc68462 (v3.0.0-124) |
| MAGNUS | github.com/omega-icl/magnus, branch main | commit ae7907c4a3f2d33c24dc1cc071dcad9c9d93b86b (2026-08-04) + patch |
| CANON | github.com/omega-icl/canon, only `src/base_opt.hpp` | commit fec71c3e5732f105ace8a27b2c64fc19a0203a86 |
| OpenBLAS | OpenBLAS-0.3.34-x64.zip from the GitHub release | sha256 e9cb6134541f36c27346d5fc5995652f060fba227cebbbabcbda5a5a44d7c76b |
| Armadillo | armadillo-15.0.3.tar.xz from SourceForge (headers only) | sha256 9f55ec10f0a91fb6479ab4ed2b37a52445aee917706a238d170b5220c022fe43 |
| Boost | boost_1_90_0.zip from archives.boost.io (headers only) | sha256 bdc79f179d1a4a60c10fe764172946d0eeafad65e576a8703c4d89d49949973c |

MAGNUS has no version-4.0 branch; its `main` is the MC++ v4 line (`import pymc`). Only one
CANON header is needed, so CANON is fetched as a single file at its commit. All pins live
at the top of `setup.py`.

## The patch: two bugs in NSFeas (nsfeas.hpp)

Both are documented in https://github.com/Kai360-dot/nested-sampler (commit 48a5429, README
section "Deliberate deviations") and are present at MAGNUS commit ae7907c.

1. Misattributed values after a failed evaluation, in single-threaded mode (the mode that
   Python callbacks require). The sequential proposal loop skipped a failed proposal without
   storing a value, so the values of all following proposals of the batch were paired with
   the wrong points and the last value was lost. Points could enter the live set with a
   criterion value belonging to another point, including points whose own evaluation had
   failed. `tests/failed_eval_alignment.py` shows it: unpatched, 114 of 300 live points
   carried a foreign value and 18 lay in the region where evaluation always fails; patched,
   0 and 0. Fix: a per-proposal failure flag (`_vERRSAM`) and a placeholder value keep
   `_vVALSAM` aligned with `_vCONSAM`; flagged proposals are skipped when the live set is
   updated, as the parallel path already does with `veval_err()`. Applied to both the
   feasibility and the likelihood loop. Side effect: failed proposals now count in
   EVALUATIONS, as in the parallel path.
2. Inverted wall-time check: `_terminate` returned "interrupt" while the elapsed time was
   still below MAXCPU, so any MAXCPU > 0 stopped the sampler at iteration 0. Fix: `<`
   becomes `>=`. `tests/maxcpu.py` shows it.

Without failed evaluations and without MAXCPU the patched sampler produces exactly the same
points as the unpatched one (checked with the upstream `nsfeas_tutorial.py` and with the
full `cyclic_cascade.ipynb`).

## Manual steps (what setup.py does)

Use an "x64 Native Tools Command Prompt for VS" so that cl, cmake and ninja are on PATH.
`%T%` stands for this folder, `%W%` for a work folder outside the repository.

```
git clone https://github.com/omega-icl/mcpp.git %W%\mcpp
git -C %W%\mcpp -c advice.detachedHead=false checkout 7b70a95535fb49cffbab7890e63accf7c4a7b5b1
git -C %W%\mcpp submodule update --init --recursive
git clone https://github.com/omega-icl/magnus.git %W%\magnus
git -C %W%\magnus -c advice.detachedHead=false checkout ae7907c4a3f2d33c24dc1cc071dcad9c9d93b86b
git -C %W%\magnus apply --check %T%\patches\0001-nsfeas-failed-proposal-alignment-and-maxcpu.patch
git -C %W%\magnus apply %T%\patches\0001-nsfeas-failed-proposal-alignment-and-maxcpu.patch

mkdir %W%\deps %W%\deps\canon %W%\deps\openblas %W%\deps\armadillo
curl -L -o %W%\deps\canon\base_opt.hpp https://raw.githubusercontent.com/omega-icl/canon/fec71c3e5732f105ace8a27b2c64fc19a0203a86/src/base_opt.hpp
curl -L -o %W%\deps\openblas.zip https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34-x64.zip
tar -xf %W%\deps\openblas.zip -C %W%\deps\openblas
curl -L -o %W%\deps\armadillo.tar.xz https://sourceforge.net/projects/arma/files/armadillo-15.0.3.tar.xz/download
tar -xf %W%\deps\armadillo.tar.xz -C %W%\deps\armadillo --strip-components=1
curl -L -o %W%\deps\boost.zip https://archives.boost.io/release/1.90.0/source/boost_1_90_0.zip
tar -xf %W%\deps\boost.zip -C %W%\deps boost_1_90_0/boost
certutil -hashfile %W%\deps\openblas.zip SHA256       (compare with the table)

cmake -S %T% -B %W%\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DOMEGA_WORK_DIR=%W% -DPython_EXECUTABLE=<base python.exe of the venv>
cmake --build %W%\build
cmake --install %W%\build --prefix <venv>\Lib\site-packages
```

The base interpreter of a venv is printed by
`<venv>\Scripts\python.exe -c "import sys; print(sys.base_prefix)"`.

## Notes

- The modules are tied to the CPython minor version they were built for (`cp314` for
  Python 3.14). After moving the venv to another minor version, rerun `setup.py --install`.
- `libopenblas.dll` must stay next to `magnus.*.pyd`.
- No MC++ source is modified. MSVC rejects the constant `0./0.` that `ffunc.hpp` uses to
  initialise FFVar unless `/fp:strict` is set, which `CMakeLists.txt` does. The alternative
  is a 3-line change to `std::numeric_limits<double>::quiet_NaN()`.
- Results are not bit-identical to a Linux build: `arma::randn` / `randu` use the C++
  standard library distributions, which differ between MSVC and GCC. Evaluation counts and
  feasible fractions agree to within a few percent.
- Only the NSFeas slice is built. Parameter estimation, experiment design and model
  discrimination (the rest of MAGNUS) need CRONOS and CANON with SUNDIALS, SNOPT, Gurobi and
  GAMS and are not part of this.
- Inline static data of MC++ (`FFExpr::options` and friends) is one shared instance across
  modules on Linux but one copy per `.pyd` on Windows. NSFeas does not depend on it.
