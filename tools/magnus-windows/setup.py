"""Build pymc (MC++ 4.0) and magnus.NSFeas natively on Windows from pinned, patched sources.

Produces two CPython extension modules, pymc and a reduced magnus (BaseSampling, BaseFeas,
BaseMBDOE, NSFeas), plus libopenblas.dll, built with the Microsoft compiler against the
CPython that the venv running this script is based on.

What it does, in order (every step is skipped when already done):
  1. clones omega-icl/mcpp and omega-icl/magnus at the pinned commits (plus pybind11 submodule)
  2. applies patches/0001-nsfeas-failed-proposal-alignment-and-maxcpu.patch to magnus
  3. downloads OpenBLAS, Armadillo, Boost headers and one CANON header, verifying sha256
  4. configures and builds with CMake + Ninja from the Visual Studio Build Tools
  5. with --install: copies the three files into the site-packages of the running interpreter
  6. runs the verification scripts in tests/

Usage (run with the Python of the venv that should receive the modules):
    .venv\\Scripts\\python.exe tools\\magnus-windows\\setup.py            # build only
    .venv\\Scripts\\python.exe tools\\magnus-windows\\setup.py --install  # build and install

Works from any prompt: the Visual Studio environment is located with vswhere. Requirements:
git, Visual Studio Build Tools with the workload "Desktop development with C++", and a
CPython install with headers (the pymanager install has them). Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- pinned sources -----------------------------------------------------------------------
MCPP_URL = "https://github.com/omega-icl/mcpp.git"
MCPP_COMMIT = "7b70a95535fb49cffbab7890e63accf7c4a7b5b1"  # branch version-4.0, 2025-11-21
MAGNUS_URL = "https://github.com/omega-icl/magnus.git"
MAGNUS_COMMIT = "ae7907c4a3f2d33c24dc1cc071dcad9c9d93b86b"  # branch main, 2026-08-04
PYBIND11_COMMIT = "eef24d6172f4a46bcc68f91d3e1ceba71cc68462"  # mcpp submodule, v3.0.0-124
CANON_COMMIT = "fec71c3e5732f105ace8a27b2c64fc19a0203a86"  # branch master (4.1)
PATCH = HERE / "patches" / "0001-nsfeas-failed-proposal-alignment-and-maxcpu.patch"

# --- pinned downloads ---------------------------------------------------------------------
DOWNLOADS = {
    "openblas.zip": (
        "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34-x64.zip",
        "e9cb6134541f36c27346d5fc5995652f060fba227cebbbabcbda5a5a44d7c76b",
    ),
    "armadillo.tar.xz": (
        "https://sourceforge.net/projects/arma/files/armadillo-15.0.3.tar.xz/download",
        "9f55ec10f0a91fb6479ab4ed2b37a52445aee917706a238d170b5220c022fe43",
    ),
    "boost.zip": (
        "https://archives.boost.io/release/1.90.0/source/boost_1_90_0.zip",
        "bdc79f179d1a4a60c10fe764172946d0eeafad65e576a8703c4d89d49949973c",
    ),
    "canon/base_opt.hpp": (
        f"https://raw.githubusercontent.com/omega-icl/canon/{CANON_COMMIT}/src/base_opt.hpp",
        "d351615d5518a908789ffea6be6fa5d20dd92fe7a7e545f078a42691ae8ff9c5",
    ),
}


def log(msg: str) -> None:
    print(f"[magnus-windows] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[magnus-windows] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(args, *, cwd=None, env=None, quiet=False) -> str:
    """Run a command, echo it, fail loudly, return its stdout."""
    args = [str(a) for a in args]
    # CreateProcess searches the parent's PATH, not env["PATH"]: resolve the executable ourselves
    exe = shutil.which(args[0], path=(env or os.environ).get("PATH"))
    if exe is None:
        fail(f"{args[0]} not found on PATH")
    args[0] = exe
    log("$ " + " ".join(args))
    proc = subprocess.run(args, cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not quiet or proc.returncode:
        sys.stdout.write(proc.stdout)
    if proc.returncode:
        fail(f"command failed with exit code {proc.returncode}")
    return proc.stdout


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- toolchain ----------------------------------------------------------------------------
def msvc_env() -> dict:
    """Environment with cl, cmake and ninja on PATH, via vcvars64.bat when needed."""
    env = dict(os.environ)
    if all(shutil.which(t, path=env["PATH"]) for t in ("cl", "cmake", "ninja")):
        log("compiler already on PATH (developer prompt)")
        return env
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        fail("Visual Studio Build Tools not found (no vswhere.exe). Install them with the "
             "workload 'Desktop development with C++'.")
    out = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                          "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                          "-property", "installationPath"],
                         text=True, stdout=subprocess.PIPE).stdout.strip()
    if not out:
        fail("no Visual Studio installation with the C++ x64 tools found. Add the workload "
             "'Desktop development with C++' in the Visual Studio Installer.")
    vcvars = Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        fail(f"{vcvars} not found")
    log(f"loading compiler environment from {vcvars}")
    proc = subprocess.run(f'cmd.exe /s /c ""{vcvars}" && set"', text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env={**env, "VSCMD_SKIP_SENDTELEMETRY": "1"})
    if proc.returncode:
        sys.stdout.write(proc.stdout)
        fail("vcvars64.bat failed")
    for line in proc.stdout.splitlines():
        m = re.match(r"^([^=]+)=(.*)$", line)
        if m:
            env[m.group(1)] = m.group(2)
    missing = [t for t in ("cl", "cmake", "ninja") if not shutil.which(t, path=env["PATH"])]
    if missing:
        fail(f"{', '.join(missing)} still not on PATH after vcvars64.bat; is the C++ workload "
             "complete (MSVC, Windows SDK, CMake and Ninja components)?")
    return env


def base_python() -> Path:
    """The CPython that the running interpreter (venv or not) is based on, with headers."""
    exe = getattr(sys, "_base_executable", None) or str(Path(sys.base_prefix) / "python.exe")
    exe = Path(exe)
    if not exe.exists():
        fail(f"base interpreter {exe} not found")
    if not (Path(sys.base_prefix) / "include" / "Python.h").exists():
        fail(f"{sys.base_prefix} has no C headers (include/Python.h). Use a CPython install "
             "that ships them, e.g. the Python install manager or python.org installer.")
    return exe


# --- sources ------------------------------------------------------------------------------
def git_clone_pinned(url: str, commit: str, dest: Path, *, submodules: bool, env) -> None:
    if (dest / ".git").exists():
        head = run(["git", "-C", dest, "rev-parse", "HEAD"], env=env, quiet=True).strip()
        if head != commit:
            log(f"{dest.name}: at {head[:9]}, switching to {commit[:9]}")
            run(["git", "-C", dest, "fetch", "--quiet", "origin"], env=env)
            run(["git", "-C", dest, "-c", "advice.detachedHead=false", "checkout", "--quiet",
                 commit], env=env)
        else:
            log(f"{dest.name}: already at {commit[:9]}")
    else:
        # autocrlf=false: keep LF so that the LF patch applies and builds are byte-identical
        run(["git", "-c", "core.autocrlf=false", "clone", "--quiet", url, dest], env=env)
        run(["git", "-C", dest, "-c", "advice.detachedHead=false", "checkout", "--quiet",
             commit], env=env)
    if submodules:
        run(["git", "-C", dest, "submodule", "update", "--init", "--recursive", "--quiet"],
            env=env)


def apply_patch(magnus: Path, env) -> None:
    check_rev = subprocess.run(["git", "-C", str(magnus), "apply", "--reverse", "--check",
                                str(PATCH)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check_rev.returncode == 0:
        log("magnus: patch already applied")
        return
    dirty = run(["git", "-C", magnus, "status", "--porcelain"], env=env, quiet=True).strip()
    if dirty:
        fail(f"{magnus} has local modifications that are not our patch:\n{dirty}\n"
             "Reset it (git -C <dir> checkout -- .) or delete the folder and rerun.")
    run(["git", "-C", magnus, "apply", "--check", PATCH], env=env)
    run(["git", "-C", magnus, "apply", PATCH], env=env)
    run(["git", "-C", magnus, "diff", "--stat"], env=env)


# --- dependencies -------------------------------------------------------------------------
def fetch(name: str, deps: Path) -> Path:
    url, digest = DOWNLOADS[name]
    dest = deps / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256(dest) == digest:
        log(f"{name}: present, sha256 ok")
        return dest
    log(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "magnus-windows-setup"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, 1 << 20)
    got = sha256(tmp)
    if got != digest:
        tmp.unlink()
        fail(f"{name}: sha256 mismatch\n  expected {digest}\n  got      {got}")
    tmp.replace(dest)
    log(f"{name}: downloaded, sha256 ok")
    return dest


def extract_all(deps: Path) -> None:
    # OpenBLAS: zip with top-level bin/, include/, lib/
    if not (deps / "openblas" / "lib" / "libopenblas.lib").exists():
        log("extracting OpenBLAS")
        with zipfile.ZipFile(deps / "openblas.zip") as z:
            z.extractall(deps / "openblas")
    # Armadillo: only include/ of armadillo-15.0.3/, first path component stripped
    if not (deps / "armadillo" / "include" / "armadillo").exists():
        log("extracting Armadillo headers")
        with tarfile.open(deps / "armadillo.tar.xz") as t:
            members = []
            for m in t.getmembers():
                parts = m.name.split("/")
                if len(parts) > 1 and parts[1] == "include":
                    m.name = "/".join(parts[1:])
                    members.append(m)
            t.extractall(deps / "armadillo", members=members, filter="data")
    # Boost: only boost_1_90_0/boost/ (headers)
    if not (deps / "boost_1_90_0" / "boost" / "version.hpp").exists():
        log("extracting Boost headers (this takes a minute)")
        with zipfile.ZipFile(deps / "boost.zip") as z:
            names = [n for n in z.namelist() if n.startswith("boost_1_90_0/boost/")]
            z.extractall(deps, members=names)


# --- build --------------------------------------------------------------------------------
def build(work: Path, env, jobs: int) -> Path:
    build_dir = work / "build"
    dist = work / "dist"
    run(["cmake", "-S", HERE, "-B", build_dir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CXX_COMPILER=cl", f"-DPython_EXECUTABLE={base_python().as_posix()}",
         "-DPython_FIND_REGISTRY=NEVER", f"-DOMEGA_WORK_DIR={work.as_posix()}"], env=env)
    run(["cmake", "--build", build_dir, "-j", str(jobs)], env=env, quiet=True)
    run(["cmake", "--install", build_dir, "--prefix", dist], env=env)
    return dist


def install(work: Path, env, allow_base: bool) -> Path:
    if sys.prefix == sys.base_prefix and not allow_base:
        fail("this interpreter is not a venv; refusing to install into the base Python "
             "(pass --allow-base-python to override)")
    target = Path(sysconfig.get_paths()["purelib"])
    run(["cmake", "--install", work / "build", "--prefix", target], env=env)
    return target


def run_tests(dist: Path | None) -> None:
    env = dict(os.environ)
    if dist is not None:
        env["PYTHONPATH"] = str(dist)
    try:
        import numpy  # noqa: F401
        have_numpy = True
    except ImportError:
        have_numpy = False
    for script in ("smoke.py", "failed_eval_alignment.py", "maxcpu.py"):
        if script == "failed_eval_alignment.py" and not have_numpy:
            log("skipping failed_eval_alignment.py (numpy not installed in this interpreter)")
            continue
        log(f"test: {script}")
        proc = subprocess.run([sys.executable, str(HERE / "tests" / script)], env=env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        sys.stdout.write(proc.stdout)
        if proc.returncode:
            fail(f"{script} failed")
        if script == "failed_eval_alignment.py" and "RESULT: CONSISTENT" not in proc.stdout:
            fail("failed_eval_alignment.py did not report RESULT: CONSISTENT")


def main() -> None:
    if sys.platform != "win32":
        fail("this script builds Windows binaries with MSVC; run it with a Windows Python")
    if sys.version_info < (3, 12):
        fail("Python 3.12 or newer is needed to run this script")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_work = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "magnus-windows-build"
    ap.add_argument("--work-dir", type=Path, default=default_work,
                    help=f"folder for clones, downloads and build (default {default_work})")
    ap.add_argument("--install", action="store_true",
                    help="copy the built files into the site-packages of this interpreter")
    ap.add_argument("--allow-base-python", action="store_true",
                    help="allow --install into a non-venv interpreter")
    ap.add_argument("--jobs", type=int, default=4, help="parallel compile jobs (default 4)")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    work: Path = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    log(f"work dir: {work}")
    log(f"interpreter: {sys.executable} (base {base_python()}, {sys.version.split()[0]})")
    if not shutil.which("git"):
        fail("git not found on PATH")
    env = msvc_env()

    git_clone_pinned(MCPP_URL, MCPP_COMMIT, work / "mcpp", submodules=True, env=env)
    pb = run(["git", "-C", work / "mcpp" / "extern" / "pybind11", "rev-parse", "HEAD"],
             env=env, quiet=True).strip()
    if pb != PYBIND11_COMMIT:
        fail(f"pybind11 submodule at {pb}, expected {PYBIND11_COMMIT}")
    git_clone_pinned(MAGNUS_URL, MAGNUS_COMMIT, work / "magnus", submodules=False, env=env)
    apply_patch(work / "magnus", env)

    deps = work / "deps"
    for name in DOWNLOADS:
        fetch(name, deps)
    extract_all(deps)

    dist = build(work, env, args.jobs)
    log(f"built: {', '.join(p.name for p in sorted(dist.iterdir()))}")

    if args.install:
        target = install(work, env, args.allow_base_python)
        log(f"installed into {target}")
        if not args.skip_tests:
            run_tests(None)
    else:
        if not args.skip_tests:
            run_tests(dist)
        log("not installed (no --install). To install into this interpreter's site-packages:")
        log(f"  {sys.executable} {Path(__file__).resolve()} --install")
        log(f"or copy the three files from {dist} into any CPython "
            f"{sys.version_info.major}.{sys.version_info.minor} x64 site-packages.")
    log("done")


if __name__ == "__main__":
    main()
